"""Stage 13：共享疾病–对照轴与各病种特异残差的细胞类型分解。

1. 细胞类型参照：两套单细胞数据只用对照供体的 pseudobulk（供体 × 细胞类型），log CPM；
   每个细胞类型取特异性最高的 top-n 标志基因（在该细胞类型出现的每个数据集中都须排进 top-2n，且表达 ≥ 阈值）。
2. 逐基因对比（样本内秩，Hedges g，按单元计算后跨单元取平均并记录符号一致性）：
   SHARED            非 DKD 病人 vs 对照
   DKD_vs_CTRL       DKD vs 对照
   <D>_vs_PAT        病种 D vs 同单元其余病人（D 含 DKD）
   <D>_vs_PAT_adj    同上，但先在病人内对每个基因回归掉外部共享轴分数（stage 11 同法、留一单元估计）
3. 富集：每个对比 × 细胞类型，标志基因效应均值相对全体基因的 z 分数（基因置换零分布）。
4. 单细胞验证：DKD 特异（adj）排名前列的基因，在被指认细胞类型的单细胞 DKD vs 对照 pseudobulk 中的方向一致率。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import NON_DISEASE, full_ranks, injury_axis, injury_score  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
CT = CFG["celltype"]
OUT = ROOT / "results" / "13_celltype"
SC = ROOT / CFG["old_project_processed"] / "scrna"


def hedges_g(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / (sd + 1e-6)


def reference() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """返回 细胞类型 → 标志基因集合；以及每个数据集的 DKD vs 对照 细胞类型 log2FC（用于验证）。"""
    spec, lfc = {}, []
    for ds in ("GSE131882", "GSE209781"):
        m = pd.read_csv(SC / f"pseudobulk_{ds}_meta.tsv", sep="\t")
        x = pd.read_csv(SC / f"pseudobulk_{ds}.tsv.gz", sep="\t", index_col=0)
        m = m[m.cell_type.isin(CT["keep"]) & (m.n_cells >= CT["min_cells"])]
        cpm = np.log2(x[m.key] / x[m.key].sum() * 1e6 + 1)
        ctrl = m[m.group == "Control"]
        mean_ct = pd.DataFrame({c: cpm[g.key].mean(1) for c, g in ctrl.groupby("cell_type")})
        z = mean_ct.sub(mean_ct.mean(1), axis=0).div(mean_ct.std(1) + 1e-6, axis=0)
        spec[ds] = (z, mean_ct)
        for c, g in m.groupby("cell_type"):
            a, b = cpm[g[g.group == "DKD"].key], cpm[g[g.group == "Control"].key]
            if a.shape[1] >= 2 and b.shape[1] >= 2:
                lfc.append(pd.DataFrame({"dataset": ds, "cell_type": c, "gene": cpm.index,
                                         "log2fc": (a.mean(1) - b.mean(1)).values}))
    markers = {}
    n = CT["n_markers"]
    for c in CT["keep"]:
        cand = None
        for ds, (z, mean_ct) in spec.items():
            if c not in z:
                continue
            ok = z[c][(mean_ct[c] >= CT["min_log2cpm"])].nlargest(2 * n).index
            cand = set(ok) if cand is None else cand & set(ok)
        if cand:
            zsum = sum(spec[ds][0][c].reindex(list(cand)).fillna(0) for ds in spec if c in spec[ds][0])
            markers[c] = zsum.nlargest(n).index
    return markers, pd.concat(lfc)


def unit_contrasts(u: str, idx: pd.Index, s: pd.DataFrame, inj: pd.Series | None) -> dict[str, pd.Series]:
    R = full_ranks(idx, s).dropna(axis=1)
    d = s.loc[idx, "diagnosis"].values
    ctrl, pat = d == "CONTROL", ~np.isin(d, list(NON_DISEASE - {"DKD"}))
    out = {}
    X = R.values
    if ctrl.sum() >= 5 and (pat & (d != "DKD")).sum() >= 5:
        out["SHARED"] = hedges_g(X[pat & (d != "DKD")], X[ctrl])
    if ctrl.sum() >= 5 and (d == "DKD").sum() >= 5:
        out["DKD_vs_CTRL"] = hedges_g(X[d == "DKD"], X[ctrl])
    Xadj = None
    if inj is not None and pat.sum() >= 10:
        z = inj.reindex(idx).values[pat]
        Z = np.c_[np.ones(pat.sum()), z]
        beta = np.linalg.lstsq(Z, X[pat], rcond=None)[0]
        Xadj = X[pat] - Z @ beta
    for dz in sorted(set(d[pat])):
        k = d[pat] == dz
        if k.sum() >= CT["min_group"] and (~k).sum() >= CT["min_group"]:
            out[f"{dz}_vs_PAT"] = hedges_g(X[pat][k], X[pat][~k])
            if Xadj is not None:
                out[f"{dz}_vs_PAT_adj"] = hedges_g(Xadj[k], Xadj[~k])
    return {k: pd.Series(v, index=R.columns) for k, v in out.items()}


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    from importlib import import_module
    s, units = import_module("11_injury_axis").load_samples()
    markers, sc_lfc = reference()
    pd.DataFrame([(c, g) for c, gs in markers.items() for g in gs], columns=["cell_type", "gene"]).to_csv(
        OUT / "celltype_markers.tsv", sep="\t", index=False)

    effects = []   # 长表：compartment unit contrast gene g
    for comp, spec in CFG["injury"]["compartments"].items():
        for u in CT["units"][comp]:
            src = [units[x] for x in spec["source"] if x != u]
            axis = injury_axis(src, s)
            R = full_ranks(units[u], s).dropna(axis=1)
            inj = injury_score(R, axis)
            for con, g in unit_contrasts(u, units[u], s, inj).items():
                effects.append(pd.DataFrame({"compartment": comp, "unit": u, "contrast": con,
                                             "gene": g.index, "g": g.values}))
    E = pd.concat(effects)
    E.to_csv(OUT / "gene_effects_by_unit.tsv.gz", sep="\t", index=False)
    meta = (E.groupby(["compartment", "contrast", "gene"]).g
            .agg(g_mean="mean", n_units="size", sign_consistency=lambda x: abs(np.sign(x).mean())).reset_index())
    meta.to_csv(OUT / "gene_effects_meta.tsv.gz", sep="\t", index=False)

    rng = np.random.default_rng(CFG["seed"])
    enr = []
    for (comp, con, unit), g in pd.concat([E.assign(level="unit"),
                                           meta.rename(columns={"g_mean": "g"}).assign(unit="META")]
                                          ).groupby(["compartment", "contrast", "unit"]):
        v = g.set_index("gene").g.dropna()
        for c, mk in markers.items():
            mk = v.index.intersection(mk)
            if len(mk) < 10:
                continue
            obs = v[mk].mean()
            null = np.array([v.values[rng.choice(len(v), len(mk), replace=False)].mean()
                             for _ in range(CT["n_perm"])])
            enr.append({"compartment": comp, "contrast": con, "unit": unit, "cell_type": c,
                        "n_markers": len(mk), "mean_g": obs, "z": (obs - null.mean()) / null.std()})
    enr = pd.DataFrame(enr)
    enr.to_csv(OUT / "celltype_enrichment.tsv", sep="\t", index=False)

    # 单细胞验证：DKD 特异（adj，meta）前 top 基因中，属于某细胞类型标志的，在该细胞类型单细胞 DKD vs 对照中的方向一致率
    val = []
    for comp in CFG["injury"]["compartments"]:
        m = meta[(meta.compartment == comp) & (meta.contrast == "DKD_vs_PAT_adj")].set_index("gene")
        if m.empty:
            continue
        top = m[m.sign_consistency == 1].g_mean.abs().nlargest(CT["n_top_validate"]).index
        for (ds, c), lf in sc_lfc.groupby(["dataset", "cell_type"]):
            lf = lf.set_index("gene").log2fc
            gg = top.intersection(lf.index)
            gg = gg[lf[gg].abs() > 0.25]
            if len(gg) < 10:
                continue
            agree = float((np.sign(lf[gg]) == np.sign(m.g_mean[gg])).mean())
            bg = m.g_mean.reindex(lf.index).dropna()
            bg = bg[lf[bg.index].abs() > 0.25]
            base = float((np.sign(lf[bg.index]) == np.sign(bg)).mean())
            val.append({"compartment": comp, "dataset": ds, "cell_type": c, "n_genes": len(gg),
                        "sign_agreement_top": agree, "sign_agreement_background": base})
    pd.DataFrame(val).to_csv(OUT / "sc_validation.tsv", sep="\t", index=False)
    write_provenance("13_celltype", [SC / "pseudobulk_GSE131882.tsv.gz", SC / "pseudobulk_GSE209781.tsv.gz"],
                     [OUT / f for f in ("celltype_markers.tsv", "gene_effects_meta.tsv.gz",
                                        "celltype_enrichment.tsv", "sc_validation.tsv")], CFG["seed"])
    piv = enr[enr.unit == "META"].pivot_table(index=["compartment", "contrast"], columns="cell_type", values="z")
    print(piv.round(1).to_string())
    print(pd.DataFrame(val).round(2).to_string())


if __name__ == "__main__":
    main()
