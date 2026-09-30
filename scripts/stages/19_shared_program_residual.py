"""Stage 19e：病种特异残差图谱 —— 扣除 SCP 后每个病种 vs 其余病人的基因与细胞程序，并在独立数据中检验。

发现集（config shared_program.residual.units；不含任何独立队列）：单元内病人（含 DKD）上，每个基因先对 SCP 分数
（来源单元用去掉本单元后重估的核心集）做线性回归取残差，再算 病种 D vs 同单元其余病人 的 Hedges g（adj）；
同时给出未调整的 g（raw）与 D 的 SCP 分数差。跨单元 DL 随机效应合并。
细胞程序：KPMP snRNA / scRNA 标志基因上 adj g_re 的均值 z（基因置换）。
一致性：与 stage 13 <D>_vs_PAT_adj（外部共享轴调整）与 stage 18 RRG-ID 各 head 基因总效应的 Spearman。
独立检验：每病种签名 = adj g_re 上 / 下调各 sig_n 个（跨单元同号）；独立队列的病人内 AUROC、SCP 调整 AUROC、
随机签名百分位；并列报告 SCP 分数本身在同一对比中的 AUROC。
产出 results/19_shared_program/residual/
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import adjusted_auroc, full_ranks  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp import auc_vs_null, hedges, patient_mask, random_set_null, re_meta, signed_score  # noqa: E402
from lib.scp_core import OUT, SP, load_core, loo_sets  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

CFG = load_config()
RS = SP["residual"]
O = OUT / "residual"


def unit_residual(comp, u, s, units) -> list[pd.DataFrame]:
    R = full_ranks(units[u], s).dropna(axis=1)
    d = s.loc[R.index, "diagnosis"].to_numpy()
    pat = patient_mask(d, include_dkd=True)
    up, dn = loo_sets(comp, u, s, units)
    z = signed_score(R, up, dn).to_numpy()[pat]
    X = R.to_numpy()[pat]
    Z = np.c_[np.ones(len(z)), z]
    Xa = X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    dp = d[pat]
    out = []
    for dz in RS["diseases"]:
        k = dp == dz
        if k.sum() < RS["min_group"] or (~k).sum() < RS["min_group"]:
            continue
        ga, va = hedges(Xa[k], Xa[~k])
        gr, _ = hedges(X[k], X[~k])
        gs, _ = hedges(z[k, None], z[~k, None])
        out.append(pd.DataFrame({"compartment": comp, "unit": u, "disease": dz, "n": int(k.sum()), "n_rest": int((~k).sum()),
                                 "gene": R.columns, "g_adj": ga, "v_adj": va, "g_raw": gr, "scp_g": float(gs[0])}))
    return out


def meta_by_disease(E: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comp, dz), g in E.groupby(["compartment", "disease"]):
        G = g.pivot(index="gene", columns="unit", values="g_adj")
        V = g.pivot(index="gene", columns="unit", values="v_adj")
        keep = G.notna().all(1)
        M = re_meta(G[keep], V[keep])
        Gr = g.pivot(index="gene", columns="unit", values="g_raw")[keep].mean(1)
        rows.append(M.assign(compartment=comp, disease=dz, g_raw_mean=Gr, n_units=G.shape[1],
                             n_patients=int(g.drop_duplicates("unit").n.sum())).rename_axis("gene").reset_index())
    return pd.concat(rows)


def signatures(M: pd.DataFrame) -> dict[tuple[str, str], tuple[list, list]]:
    out = {}
    for (comp, dz), g in M.groupby(["compartment", "disease"]):
        g = g[g.sign_frac == 1.0].set_index("gene").g_re
        out[(comp, dz)] = (g.nlargest(RS["sig_n"]).index.tolist(), g.nsmallest(RS["sig_n"]).index.tolist())
    return out


def programs(M: pd.DataFrame, rng) -> pd.DataFrame:
    mk = pd.read_csv(ROOT / "results/14_kpmp/markers_kpmp.tsv", sep="\t")
    mk = mk[mk.reference.isin(["snRNA", "scRNA"])]
    rows = []
    for (comp, dz), g in M.groupby(["compartment", "disease"]):
        v = g.set_index("gene").g_re
        for (ref, ct), m in mk.groupby(["reference", "cell_type"]):
            gg = v.index.intersection(m.gene)
            if len(gg) < 10:
                continue
            obs = v[gg].mean()
            null = np.array([v.to_numpy()[rng.choice(len(v), len(gg), replace=False)].mean() for _ in range(RS["n_perm"])])
            rows.append({"compartment": comp, "disease": dz, "reference": ref, "cell_type": ct, "n_markers": len(gg),
                         "mean_g_adj": obs, "z": (obs - null.mean()) / null.std()})
    return pd.DataFrame(rows)


def consistency(M: pd.DataFrame) -> pd.DataFrame:
    s13 = pd.read_csv(ROOT / "results/13_celltype/gene_effects_meta.tsv.gz", sep="\t")
    s18 = pd.read_csv(ROOT / "results/18_interpret/weights/gene_effects.tsv.gz", sep="\t")
    rows = []
    for (comp, dz), g in M.groupby(["compartment", "disease"]):
        v = g.set_index("gene")
        a = s13[(s13.compartment == comp) & (s13.contrast == f"{dz}_vs_PAT_adj")].set_index("gene").g_mean
        b = s18[(s18.compartment == comp) & (s18["head"] == dz)].set_index("gene").e_full
        r = {"compartment": comp, "disease": dz}
        for nm, ref in (("stage13_adj", a), ("stage18_head", b)):
            gi = v.index.intersection(ref.index)
            r[f"spearman_{nm}"] = spearmanr(v.g_re[gi], ref[gi])[0] if len(gi) > 50 else np.nan
            top = v.g_re.abs().nlargest(100).index.intersection(ref.index)
            r[f"top100_sign_agree_{nm}"] = float((np.sign(v.g_re[top]) == np.sign(ref[top])).mean()) if len(top) else np.nan
        r["spearman_raw_vs_scp"] = spearmanr(v.g_raw_mean, pd.read_csv(OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t")
                                             .set_index("gene").g_re.reindex(v.index), nan_policy="omit")[0]
        rows.append(r)
    return pd.DataFrame(rows)


def independent_tests(sigs, s, units, rng) -> pd.DataFrame:
    S19d = import_module("19_shared_program_replicate")
    rows = []
    cache = {}
    for cohort, comp, pos, negs in RS["tests"]:
        if cohort not in cache:
            spec = SP["replicate"]["cohorts"][cohort]
            cache[cohort] = S19d.load_cohort(cohort, spec, s, units)
        R, meta = cache[cohort]
        d = meta.diagnosis.to_numpy()
        m = np.isin(d, [pos] + negs)
        y = (d[m] == pos).astype(int)
        if y.sum() < RS["min_test_n"] or (1 - y).sum() < RS["min_test_n"] or (comp, pos) not in sigs:
            rows.append({"cohort": cohort, "compartment": comp, "disease": pos, "negatives": "+".join(negs),
                         "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "status": "not_testable"})
            continue
        up, dn = sigs[(comp, pos)]
        up, dn = [x for x in up if x in R.columns], [x for x in dn if x in R.columns]
        sc = signed_score(R, up, dn).to_numpy()[m]
        cu, cd = load_core(SP["replicate"]["cohorts"][cohort]["core"])
        scp = signed_score(R, cu, cd).to_numpy()[m]
        null = random_set_null(R.iloc[np.flatnonzero(m)], len(up), len(dn), SP["replicate"]["n_random_sets"], rng)
        r = auc_vs_null(y, sc, null)
        adj = adjusted_auroc(y, sc, scp)
        null_adj = np.array([adjusted_auroc(y, z, scp) for z in null])
        small = min(y.sum(), (1 - y).sum()) < 5
        rows.append({"cohort": cohort, "compartment": comp, "disease": pos, "negatives": "+".join(negs),
                     "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_up": len(up), "n_down": len(dn), **r,
                     "auroc_scp_adjusted": adj, "pct_vs_random_adjusted": float((null_adj < adj).mean()),
                     "auroc_scp_itself": float(auc_rows(y, scp[None])[0]),
                     "status": "descriptive" if small else ("replicated" if r["pct_vs_random"] >= RS["replicate_pct"]
                                                            and r["auroc"] > 0.5 else "not_replicated")})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, units = import_module("11_injury_axis").load_samples()
    E = pd.concat([x for comp, us in RS["units"].items() for u in us for x in unit_residual(comp, u, s, units)])
    E.to_csv(O / "unit_effects.tsv.gz", sep="\t", index=False)
    scp_shift = E.drop_duplicates(["compartment", "unit", "disease"])[["compartment", "unit", "disease", "n", "n_rest", "scp_g"]]
    scp_shift.to_csv(O / "disease_scp_shift.tsv", sep="\t", index=False)
    M = meta_by_disease(E)
    M.to_csv(O / "residual_meta.tsv.gz", sep="\t", index=False)
    sigs = signatures(M)
    top = []
    for (comp, dz), (up, dn) in sigs.items():
        g = M[(M.compartment == comp) & (M.disease == dz)].set_index("gene")
        for dr, gl in (("up", up[:RS["n_top_report"]]), ("down", dn[:RS["n_top_report"]])):
            top += [{"compartment": comp, "disease": dz, "direction": dr, "rank": i + 1, "gene": x,
                     "g_adj_re": g.g_re[x], "g_raw_mean": g.g_raw_mean[x], "n_units": g.n_units[x], "I2": g.I2[x]}
                    for i, x in enumerate(gl)]
    pd.DataFrame(top).to_csv(O / "top_genes.tsv", sep="\t", index=False)
    pd.DataFrame([{"compartment": c, "disease": d, "direction": dr, "gene": x}
                  for (c, d), (u, dn) in sigs.items() for dr, gl in (("up", u), ("down", dn)) for x in gl]
                 ).to_csv(O / "signatures.tsv", sep="\t", index=False)
    P = programs(M, rng)
    P.to_csv(O / "cell_programs.tsv", sep="\t", index=False)
    C = consistency(M)
    C.to_csv(O / "consistency.tsv", sep="\t", index=False)
    T = independent_tests(sigs, s, units, rng)
    T.to_csv(O / "independent_tests.tsv", sep="\t", index=False)
    outs = [O / f for f in ("residual_meta.tsv.gz", "top_genes.tsv", "signatures.tsv", "cell_programs.tsv",
                            "consistency.tsv", "independent_tests.tsv", "disease_scp_shift.tsv")]
    write_provenance("19_shared_program/residual", [OUT / "core/core_genes.tsv", ROOT / "results/14_kpmp/markers_kpmp.tsv",
                                                    ROOT / "results/13_celltype/gene_effects_meta.tsv.gz",
                                                    ROOT / "results/18_interpret/weights/gene_effects.tsv.gz"],
                     outs, CFG["seed"], {"units": RS["units"], "sig_n": RS["sig_n"]})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(scp_shift.round(2).to_string())
        print(C.round(3).to_string())
        t = pd.DataFrame(top)
        for (c, d), g in t.groupby(["compartment", "disease"]):
            print(c, d, "UP", g[g.direction == "up"].gene.head(8).tolist(), "DOWN", g[g.direction == "down"].gene.head(8).tolist())
        p = P[P.reference == "snRNA"].pivot_table(index="cell_type", columns=["compartment", "disease"], values="z")
        print(p.round(1).to_string())
        print(T.round(3).to_string())


if __name__ == "__main__":
    main()
