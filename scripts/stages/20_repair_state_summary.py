"""Stage 20h：主图输入（figdata/，只从 20a–20g 的表整理，不重算模型）与正文关键数字；顶层 PROVENANCE 汇总子目录。
产出 results/20_repair_state/figdata/ 与 results/20_repair_state/PROVENANCE.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repair_state import CFG, OUT  # noqa: E402
from lib.repro import set_global_seed, write_provenance  # noqa: E402

F = OUT / "figdata"
SUBS = ["programs", "concord", "decomp", "donors", "bulk", "clinical", "genes"]


def rd(sub, f, **kw):
    return pd.read_csv(OUT / sub / f, sep="\t", **kw)


PT_TAL_BLOCKS = ["repair_state_fraction", "PT_TAL_normal_cell_profile", "PT_TAL_repair_cell_profile",
                 "PT_TAL_other_altered_profile"]


def pt_tal_shares(blk: pd.DataFrame) -> list[dict]:
    """PT+TAL 状态差（4 个 PT/TAL 块之和）中各块的份额；修复失败 = 占比 + 修复失败细胞谱。"""
    rows = []
    for (ds, ct), g in blk.groupby(["dataset", "contrast"]):
        d = g.set_index("component").delta
        tot = d.reindex(PT_TAL_BLOCKS).sum()
        for b in PT_TAL_BLOCKS:
            rows.append({"item": f"pt_tal_share|{ds}|{ct}|{b}", "value": d[b] / tot})
        rows.append({"item": f"pt_tal_share|{ds}|{ct}|repair_attributable",
                     "value": (d["repair_state_fraction"] + d["PT_TAL_repair_cell_profile"]) / tot})
    return rows


def covariation(ls: pd.DataFrame) -> list[dict]:
    """供体水平：修复失败占比 vs 正常细胞 SCP 分数的 Spearman（置换 P，供体为单位）。"""
    rng = np.random.default_rng(CFG["seed"])
    groups = {"REF+CKD": ["REF"] + CFG["repair_state"]["ckd"], "CKD": CFG["repair_state"]["ckd"]}
    rows = []
    for (ds, lin), g in ls.dropna(subset=["repair_frac", "scp_normal_cells"]).groupby(["dataset", "lineage"]):
        for name, cats in groups.items():
            t = g[g.category.isin(cats)]
            x = t.repair_frac.rank().to_numpy()
            y = t.scp_normal_cells.rank().to_numpy()
            rho = np.corrcoef(x, y)[0, 1]
            null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(CFG["repair_state"]["n_perm"])])
            p = (1 + (np.abs(null) >= abs(rho)).sum()) / (1 + len(null))
            rows.append({"item": f"covary|{ds}|{lin}|{name}|repair_frac_vs_normal_cell_scp", "value": rho,
                         "n": len(t), "p_perm": p})
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    F.mkdir(parents=True, exist_ok=True)
    out = {}
    # a 状态程序概况 + 关键基因在各状态的表达
    out["panel_a_program_summary.tsv"] = rd("programs", "program_summary.tsv")
    kg = rd("genes", "key_genes.tsv")
    se = rd("genes", "state_expression.tsv.gz")
    show = kg.gene[kg.is_driver].head(30).tolist() + [g for g in ["VCAM1", "HAVCR1", "PROM1", "CDH6", "CD24", "SPP1", "LCN2"]
                                                       if g not in kg.gene[kg.is_driver].head(30).tolist()]
    out["panel_a_key_gene_state_expression.tsv"] = se[se.gene.isin(show)]
    # b 逐基因一致性散点：PT / TAL 谱系 CKD vs REF log2FC vs 修复失败程序 log2FC（snRNA）
    ld = rd("concord", "lineage_de.tsv.gz")
    pr = rd("programs", "programs.tsv.gz")
    pr = pr[(pr.subset == "all") & pr.state.isin(["rfPT", "rfTAL"])][["dataset", "lineage", "gene", "log2fc", "call"]]
    b = ld[ld.contrast == "CKD_vs_REF"].merge(pr.rename(columns={"log2fc": "program_log2fc", "call": "program_call"}),
                                              on=["dataset", "lineage", "gene"])
    out["panel_b_concordance_scatter.tsv.gz"] = b
    c = rd("concord", "concordance.tsv")
    out["panel_b_concordance_stats.tsv"] = c[c.program.isin(["aPT", "frPT", "rfPT", "dPT", "cycPT", "aTAL", "frTAL", "rfTAL",
                                                             "dTAL", "cycTAL"])]
    out["panel_b_bulk_overlap.tsv"] = rd("concord", "bulk_overlap.tsv")
    # c Shapley 分解
    sc = rd("decomp", "shapley_contrast.tsv")
    out["panel_c_shapley_blocks.tsv"] = sc[~sc.component.str.startswith("factor:")]
    out["panel_c_shapley_factors.tsv"] = sc[sc.component.str.startswith("factor:")]
    # d/e 供体比例（类别箱线 + 与 SCP / eGFR 散点）
    out["panel_d_donor_table.tsv"] = rd("donors", "donor_table.tsv", dtype={"donor": str})
    out["panel_d_category_tests.tsv"] = rd("donors", "category_tests.tsv")
    out["panel_e_correlations.tsv"] = rd("donors", "correlations.tsv")
    out["panel_e_regression.tsv"] = rd("concord", "regression.tsv")
    # f bulk 跨队列
    out["panel_f_bulk_meta.tsv"] = rd("bulk", "meta_contrasts.tsv")
    out["panel_f_bulk_unit.tsv"] = rd("bulk", "unit_contrasts.tsv")
    out["panel_f_bulk_scp_relation.tsv"] = rd("bulk", "scp_relation.tsv")
    out["panel_f_bulk_resolvability.tsv"] = rd("bulk", "resolvability.tsv")
    # g 独立临床
    out["panel_g_clinical_tests.tsv"] = rd("clinical", "tests.tsv")
    out["panel_g_clinical_scores.tsv.gz"] = rd("clinical", "scores.tsv.gz")
    # h 关键基因
    out["panel_h_key_genes.tsv"] = kg
    for f, t in out.items():
        t.to_csv(F / f, sep="\t", index=False)
    # 关键数字
    kn = []
    blk = out["panel_c_shapley_blocks.tsv"]
    for _, r in blk.iterrows():
        kn.append({"item": f"shapley|{r.dataset}|{r.contrast}|{r.component}", "value": r.delta,
                   "share_of_full": r.share_of_full, "share_of_state": r.share_of_state,
                   "ci": f"{r.share_of_state_lo:.3f}–{r.share_of_state_hi:.3f}" if np.isfinite(r.share_of_state_lo) else "",
                   "p_perm": r.p_perm})
    for _, r in c[(c.gene_set == "scp_core") & c.program.isin(["rfPT", "rfTAL"])].iterrows():
        kn.append({"item": f"concord|{r.dataset}|{r.lineage}_{r.lineage_cells}|{r.contrast}|{r.program}",
                   "value": r.spearman, "p_perm": r.p_perm})
    kn += pt_tal_shares(blk) + covariation(rd("concord", "donor_lineage_scores.tsv", dtype={"donor": str}))
    pd.DataFrame(kn).to_csv(F / "key_numbers.tsv", sep="\t", index=False)
    outs = [F / f for f in out] + [F / "key_numbers.tsv"]
    write_provenance("20_repair_state/figdata", [OUT / s / "PROVENANCE.json" for s in SUBS], outs, CFG["seed"], {})
    top = {s: json.loads((OUT / s / "PROVENANCE.json").read_text()) for s in SUBS + ["figdata"]}
    (OUT / "PROVENANCE.json").write_text(json.dumps({"stage": "20_repair_state", "substages": top}, indent=1))
    print(pd.DataFrame(kn).head(80).round(3).to_string())


if __name__ == "__main__":
    main()
