"""Stage 22 S3：bulk BayesPrism 中 aPT+frPT 份额（ERCB 芯片约 0.2–0.3%）与 KPMP 实测占比（12–31%）差距的来源。

A. 参照中状态谱与正常 PT 的相似度（scRNA_TUB 参照，状态合并谱 log2(CPM+1)，BayesPrism 标志基因与全部基因）；
   标志基因中区分 aPT / frPT 与正常 PT 的基因数（|log2FC| ≥ 1），及各 bulk 平台测到其中多少。
B. 已有 bulk 结果按平台汇总（stage 20 resolvability：各单元 aPT、frPT、aPT+frPT 份额中位数与平台）。
C. 已知份额的模拟混合物：留出 n_holdout 个供体（参照只用其余供体，标志基因沿用 stage 16 缓存），
   固定小管间质组成（bulk θ_type 中位数），PT 内 aPT+frPT 占 PT 读数的份额取 shares；
   测量场景 modes（泊松计数 / 芯片样失真，全部基因或仅 ERCB 芯片基因），每场景一批 BayesPrism（与 stage 16 同一 R 脚本与参数）。
   回收率 = 估计份额（θ_state first，同 stage 20 theta_shares）对已知份额。
产出 results/22_sensitivity/S3_deconv/{profile_similarity.tsv, marker_coverage.tsv, bulk_platform_shares.tsv,
      sim_truth.tsv, sim_recovery.tsv, sim_summary.tsv, PROVENANCE.json}
用法：python scripts/stages/22_sensitivity_s3_deconv.py [--skip-bp]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.deconv_ref import OUT as DOUT, ref_dir  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
R3 = CFG["sensitivity22"]
P = R3["s3"]
O = ROOT / "results" / "22_sensitivity" / "S3_deconv"
WORK = ROOT / P["out_dir"]
RUN = import_module("16_deconv_run")


def load_ref(name: str) -> tuple[sp.csr_matrix, pd.DataFrame, pd.Index, pd.Index]:
    d = ref_dir(name, "TUB")
    tr = pd.read_parquet(d / "counts_triplet.parquet")
    rows = pd.read_csv(d / "rows.tsv", sep="\t", keep_default_na=False)
    genes = pd.Index((d / "genes.txt").read_text().split("\n")[:-1])
    X = sp.csr_matrix((tr.x, (tr.i - 1, tr.j - 1)), shape=(len(rows), len(genes)))
    mk = pd.Index((d / "markers.txt").read_text().split("\n")[:-1])
    return X, rows, genes, mk


def pooled(X, rows: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    v = np.asarray(X[np.flatnonzero(mask)].sum(0)).ravel()
    return v / max(v.sum(), 1)


def state_profiles(X, rows) -> dict[str, np.ndarray]:
    st = rows.state.to_numpy()
    prof = {"normal PT": pooled(X, rows, np.isin(st, P["normal_pt"])),
            "aPT": pooled(X, rows, st == "aPT"), "frPT": pooled(X, rows, st == "frPT"),
            "dPT": pooled(X, rows, st == "dPT"),
            "normal TAL": pooled(X, rows, np.isin(st, ["C-TAL", "C/M-TAL", "M-TAL"])),
            "aTAL": pooled(X, rows, st == "aTAL"), "frTAL": pooled(X, rows, st == "frTAL")}
    return prof


def part_a(X, rows, genes, mk) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof = state_profiles(X, rows)
    L = {k: np.log2(v * 1e6 + 1) for k, v in prof.items()}
    im = genes.get_indexer(mk.intersection(genes))
    pairs = [("aPT", "normal PT"), ("frPT", "normal PT"), ("aPT", "frPT"), ("dPT", "normal PT"),
             ("normal TAL", "normal PT"), ("aTAL", "normal TAL"), ("frTAL", "normal TAL")]
    out = []
    for a, b in pairs:
        for gs, idx in (("markers", im), ("all_expressed", np.flatnonzero((L[a] > 1) | (L[b] > 1)))):
            d = L[a][idx] - L[b][idx]
            out.append({"state_A": a, "state_B": b, "genes": gs, "n_genes": len(idx),
                        "pearson_log": float(np.corrcoef(L[a][idx], L[b][idx])[0, 1]),
                        "median_abs_log2fc": float(np.median(np.abs(d))),
                        "n_abs_log2fc_ge1": int((np.abs(d) >= 1).sum())})
    sim = pd.DataFrame(out)
    # 标志基因中区分 aPT / frPT 与正常 PT 的基因在各 bulk 平台上的覆盖
    dist = {s: set(mk.intersection(genes)[np.abs(L[s][im] - L["normal PT"][im]) >= 1]) for s in ("aPT", "frPT")}
    man = pd.read_csv(ROOT / "results/16_deconv/prepare/mixture_manifest.tsv", sep="\t")
    man = man[man.reference_compartment == "TUB"]
    cov = []
    for r in man.itertuples():
        used = RUN.bp_out(P["reference"], r.series, r.compartment) / "genes_used.txt"
        if not used.exists():
            continue
        u = set(used.read_text().split("\n")[:-1])
        cov.append({"series": r.series, "compartment": r.compartment, "technology": r.technology,
                    "markers_used": len(u), "markers_total": len(mk),
                    "aPT_vs_normal_markers_used": len(dist["aPT"] & u), "aPT_vs_normal_markers_total": len(dist["aPT"]),
                    "frPT_vs_normal_markers_used": len(dist["frPT"] & u), "frPT_vs_normal_markers_total": len(dist["frPT"])})
    return sim, pd.DataFrame(cov)


def part_b() -> pd.DataFrame:
    rv = pd.read_csv(ROOT / "results/20_repair_state/bulk/resolvability.tsv", sep="\t")
    rv = rv[rv.state.isin(["PT:aPT", "PT:frPT", "PT:rfPT"])]
    W = rv.pivot_table(index=["unit", "n"], columns="state", values="median_share").reset_index()
    platform = {"ERCB_TUB_H1": "HG-U133A (3' array)", "ERCB_TUB_H7": "HG-U133 Plus 2.0 (3' array)",
                "GSE30529-GPL571CEL": "HG-U133A 2.0 (3' array)", "GSE108112-GPL19983": "HuGene 2.1 ST array",
                "GSE133288-GPL19983": "HuGene 2.1 ST array", "GSE115857-GPL14951": "Illumina HT-12 v4 array"}
    W["platform"] = W.unit.map(platform).fillna("RNA-seq")
    return W.rename(columns={"PT:aPT": "median_aPT_share", "PT:frPT": "median_frPT_share",
                             "PT:rfPT": "median_aPT_frPT_share"})


# ---------------------------------------------------------------- C. 模拟
def holdout_donors(rows: pd.DataFrame, rng) -> list[str]:
    pt = rows[rows.type == "PT"].pivot_table(index="donor", columns="state", values="n_cells", aggfunc="sum", fill_value=0)
    norm = pt[[c for c in P["normal_pt"] if c in pt.columns]].sum(1)
    ok = (pt.aPT >= P["min_cells_state"]) & (pt.frPT >= P["min_cells_state"]) & (norm >= P["min_cells_normal"])
    return sorted(rng.choice(np.array(sorted(ok[ok].index)), P["n_holdout"], replace=False).tolist())


def write_train_reference(X, rows, genes, hold: list[str]) -> Path:
    src = ref_dir(P["reference"], "TUB")
    d = WORK / "reference" / f"{P['reference']}_TUB_holdout"
    d.mkdir(parents=True, exist_ok=True)
    keep = ~rows.donor.isin(hold).to_numpy()
    C = X[np.flatnonzero(keep)].tocoo()
    pd.DataFrame({"i": C.row.astype(np.int32) + 1, "j": C.col.astype(np.int32) + 1,
                  "x": C.data.astype(np.float64)}).to_parquet(d / "counts_triplet.parquet")
    rows[keep].to_csv(d / "rows.tsv", sep="\t", index=False)
    (d / "genes.txt").write_text("\n".join(genes) + "\n")
    (d / "markers.txt").write_text((src / "markers.txt").read_text())   # 标志基因沿用 stage 16 缓存
    return d


def composition() -> pd.Series:
    t = pd.read_csv(ROOT / "results/16_deconv/run/all/theta_type.tsv", sep="\t")
    t = t[(t.reference == P["reference"]) & (t.compartment == "TUB")]
    m = t.groupby("type").theta.median()
    return m / m.sum()


def mixtures(X, rows, genes, hold, comp, rng) -> tuple[np.ndarray, pd.DataFrame]:
    """返回 期望读数比例（样本 × 基因）与真值表。"""
    st, ty, dn = rows.state.to_numpy(), rows.type.to_numpy(), rows.donor.to_numpy()
    hmask = np.isin(dn, hold)
    wa, wf = np.array(P["apt_to_frpt"], float) / sum(P["apt_to_frpt"])
    M, truth = [], []
    for d in hold:
        m_d = dn == d
        prof = {}
        for k in comp.index:
            if k == "PT":
                continue
            msk = m_d & (ty == k)
            prof[k] = pooled(X, rows, msk if msk.any() else hmask & (ty == k))
        norm = pooled(X, rows, m_d & np.isin(st, P["normal_pt"]))
        apt, frpt = pooled(X, rows, m_d & (st == "aPT")), pooled(X, rows, m_d & (st == "frPT"))
        for s in P["shares"]:
            pt = (1 - s) * norm + s * (wa * apt + wf * frpt)
            x = comp["PT"] * pt + sum(comp[k] * prof[k] for k in prof)
            M.append(x / x.sum())
            truth.append({"sample_uid": f"{d}|s{s:.2f}", "donor": d, "true_share": s,
                          "true_aPT": s * wa, "true_frPT": s * wf})
    return np.vstack(M), pd.DataFrame(truth)


def measure(M: np.ndarray, spec: dict, rng) -> np.ndarray:
    lam = M * P["depth"]
    y = rng.poisson(lam).astype(float)
    cpm = y / y.sum(1, keepdims=True) * 1e6
    if spec["kind"] == "array":
        bias = rng.normal(0, spec["bias_sd"], size=M.shape[1])          # 逐基因探针亲和力：同一批次内所有样本共享
        v = np.log2(cpm * np.exp2(bias)[None, :] + 2.0 ** spec["floor_log2"])
        v = v + rng.normal(0, spec["noise_sd"], size=v.shape)
        cpm = np.exp2(v)                                                # stage 16 芯片线性化：2^x
    L = cpm / cpm.sum(1, keepdims=True) * 1e6
    return np.round(L).astype(np.int64)


def run_modes(X, rows, genes, hold, rng, skip_bp: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = composition()
    M, truth = mixtures(X, rows, genes, hold, comp, rng)
    refd = write_train_reference(X, rows, genes, hold)
    ercb = set(pd.read_csv(DOUT / "mixture" / f"{P['ercb_genes_from']}__TUB.tsv.gz", sep="\t", usecols=["gene"]).gene)
    jobs = []
    for mode, spec in P["modes"].items():
        g = np.flatnonzero(genes.isin(ercb)) if spec["genes"] == "ercb" else np.arange(len(genes))
        Y = measure(M[:, g] / M[:, g].sum(1, keepdims=True), spec, np.random.default_rng(rng.integers(1 << 31)))
        f = WORK / "mixture" / f"sim__{mode}.tsv.gz"
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(Y.T, index=pd.Index(genes[g], name="gene"), columns=truth.sample_uid).to_csv(f, sep="\t")
        jobs.append((mode, f, WORK / "bp" / mode))
    if not skip_bp:
        def one(j):
            mode, f, out = j
            if (out / "DONE").exists():
                return mode
            args = RUN.r_args(P["reference"], "TUB", str(f), str(out), P["n_cores"])
            args = [a if not a.startswith("ref=") else f"ref={refd}" for a in args]
            rc = RUN.run_r(args, ROOT / "logs" / "22_sensitivity" / f"s3_{mode}.log")
            if rc != 0 or not (out / "DONE").exists():
                raise RuntimeError(f"BayesPrism 失败：{mode}")
            return mode
        with ThreadPoolExecutor(len(jobs)) as ex:
            print(list(ex.map(one, jobs)))
    rec = []
    for mode, _, out in jobs:
        t = pd.read_parquet(out / "theta_state.parquet").set_index("sample_uid")
        ptc = [c for c in t.columns if c in P["normal_pt"] + ["aPT", "frPT", "dPT", "cycPT"]]
        tot = t[ptc].sum(1)
        e = pd.DataFrame({"est_aPT": t["aPT"] / tot, "est_frPT": t["frPT"] / tot})
        e["est_share"] = e.est_aPT + e.est_frPT
        rec.append(truth.merge(e, left_on="sample_uid", right_index=True).assign(mode=mode))
    return truth, pd.concat(rec)


def summarise(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, r in rec.groupby("mode"):
        for s, q in r.groupby("true_share"):
            rows.append({"mode": mode, "true_share": s, "n": len(q), "median_est_share": q.est_share.median(),
                         "median_est_aPT": q.est_aPT.median(), "median_est_frPT": q.est_frPT.median()})
        rs = r[r.true_share > 0]
        slope = np.polyfit(r.true_share, r.est_share, 1)[0]
        rows.append({"mode": mode, "true_share": "summary", "n": len(r),
                     "median_recovery_ratio": float((rs.est_share / rs.true_share).median()),
                     "slope_est_vs_true": float(slope),
                     "spearman_within_donor": float(r.groupby("donor").apply(
                         lambda q: q.true_share.corr(q.est_share, method="spearman")).median())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bp", action="store_true")
    a = ap.parse_args()
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    X, rows, genes, mk = load_ref(P["reference"])
    sim, cov = part_a(X, rows, genes, mk)
    plat = part_b()
    hold = holdout_donors(rows, rng)
    truth, rec = run_modes(X, rows, genes, hold, rng, a.skip_bp)
    summ = summarise(rec)
    outs = {"profile_similarity.tsv": sim, "marker_coverage.tsv": cov, "bulk_platform_shares.tsv": plat,
            "sim_truth.tsv": truth, "sim_recovery.tsv": rec, "sim_summary.tsv": summ}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    d = ref_dir(P["reference"], "TUB")
    write_provenance("22_sensitivity/S3_deconv",
                     [d / "counts_triplet.parquet", d / "rows.tsv", d / "markers.txt",
                      ROOT / "results/20_repair_state/bulk/resolvability.tsv",
                      ROOT / "results/16_deconv/run/all/theta_type.tsv"],
                     [O / f for f in outs], CFG["seed"], {"holdout_donors": hold, "s3": P})
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(sim.round(3).to_string()); print(cov.to_string()); print(plat.round(4).to_string())
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
