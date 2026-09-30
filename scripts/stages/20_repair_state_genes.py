"""Stage 20g：关键基因 —— SCP（TUB）核心基因中同时是修复失败小管状态驱动基因者。

驱动基因：snRNA 全部供体配对程序（stage 20a）中 状态 vs 正常 FDR < fdr 且 |lfc| ≥ lfc，且 scRNA 同号 |lfc| ≥ rep_lfc；
状态 ∈ {aPT, frPT, aTAL, frTAL}（修复失败主定义）；方向须与 SCP 核心方向一致。
每个基因报告：
  KPMP 状态特异性    供体 × 类型（PT 拆为 正常 / 修复失败 / 其他改变；TAL 同；其余按 stage 13 粗类型）均值 log2CPM 的最高类型、
                     修复失败类型相对其余全部类型最大值的 log2 比；CKD 供体组织计数中来自修复失败细胞的份额（中位数）
  KPMP 疾病变化      正常 PT / TAL 细胞内 CKD vs REF log2FC（stage 20b）；组织层 CKD vs REF Δlog2CPM 的 Shapley 份额（stage 20c）
  bulk               SCP 合并 g_re 与来源单元 g
  独立复现           每个独立队列 疾病 vs 对照 Hedges g（秩尺度）与临床 Spearman；同号且 |g| ≥ 0.5 的队列数
另附用户关注的经典损伤 / 修复标志（不论是否入核心）的同一套字段。
产出 results/20_repair_state/genes/
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_stats import log2cpm  # noqa: E402
from lib.repair_state import CFG, K, OUT, RS, STATES, category_index, donor_categories  # noqa: E402
from lib.repro import set_global_seed, write_provenance  # noqa: E402
from lib.scp import hedges  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core_table  # noqa: E402

O = OUT / "genes"
DRIVER_STATES = {"PT": ["aPT", "frPT"], "TAL": ["aTAL", "frTAL"]}
WATCH = ["VCAM1", "HAVCR1", "PROM1", "CDH6", "CD24", "SPP1", "LCN2", "CCL2", "ITGB8", "DCDC2", "SOX9", "CD44", "TPM1",
         "KIRREL1", "ITGB6", "MET", "CLDN4", "UMOD", "SLC12A1", "EGF", "LRP2", "SLC34A1"]


def type_map(pb) -> dict[str, list[str]]:
    present = set(pb.rows.fine)
    out = {}
    for c, fs in K["map_to_stage13"].items():
        for f in fs:
            if f not in present:
                continue
            if f in STATES:
                lin, st = STATES[f]
                g = "normal" if st == "normal" else "repair" if st in RS["repair_states"][lin] else "other_alt"
                key = f"{lin}_{g}"
            else:
                key = "LOH_rest" if c == "LOH" else "PT_other_alt" if c == "PT_injured" else c
            out.setdefault(key, []).append(f)
    return out


def state_specificity(pb, genes: list[str], cat) -> pd.DataFrame:
    rows, X = pb.aggregate(type_map(pb))
    gi = pb.genes.get_indexer(genes)
    ok = rows.n_cells.to_numpy() >= RS["min_cells"]
    L = log2cpm(X[np.flatnonzero(ok)])[:, gi]
    r = rows[ok].reset_index(drop=True)
    mean = pd.DataFrame(L, columns=genes).groupby(r.cell_type.values).mean().T        # 基因 × 类型
    out = pd.DataFrame(index=genes)
    out["top_type"] = mean.idxmax(1)
    for lin in ("PT", "TAL"):
        rep = f"{lin}_repair"
        others = mean.drop(columns=[rep])
        out[f"{lin}_repair_log2cpm"] = mean[rep]
        out[f"{lin}_repair_vs_max_other"] = mean[rep] - others.max(1)
        out[f"{lin}_repair_vs_{lin}_normal"] = mean[rep] - mean[f"{lin}_normal"]
    # CKD 供体：组织计数中来自修复失败细胞的份额
    ckd = category_index(cat, "CKD")
    Xc = X[:, gi].toarray()
    tot = pd.DataFrame(Xc, columns=genes).groupby(rows.donor.values).sum()
    for lin in ("PT", "TAL"):
        m = (rows.cell_type == f"{lin}_repair").to_numpy()
        rep = pd.DataFrame(Xc[m], columns=genes, index=rows.donor[m].values)
        sh = rep.reindex(tot.index).fillna(0) / tot.where(tot > 0)
        out[f"{lin}_repair_share_of_tissue_counts_CKD_median"] = sh.loc[sh.index.intersection(ckd)].median(0)
        out[f"{lin}_repair_share_of_tissue_counts_REF_median"] = sh.loc[sh.index.intersection(category_index(cat, "REF"))].median(0)
    return out.rename_axis("gene").reset_index(), mean.rename_axis("gene").reset_index()


def independent_effects(genes: list[str]) -> pd.DataFrame:
    S19 = import_module("19_shared_program_replicate")
    s, units = import_module("11_injury_axis").load_samples()
    spec_all = CFG["shared_program"]["replicate"]["cohorts"]
    rows = []
    for name in RS["bulk"]["clinical_cohorts"]:
        spec = spec_all[name]
        R, meta = S19.load_cohort(name, spec, s, units)
        g_ok = [g for g in genes if g in R.columns]
        d = meta.diagnosis.to_numpy()
        pos, ctrl = np.isin(d, spec["positive"]), d == "CONTROL"
        g, _ = hedges(R[g_ok].to_numpy()[pos], R[g_ok].to_numpy()[ctrl])
        rows.append(pd.DataFrame({"cohort": name, "test": "disease_vs_control", "gene": g_ok, "value": g,
                                  "n": int(pos.sum() + ctrl.sum())}))
        for test, mk, var in S19.clinical(name, meta):
            if mk.sum() < 8:
                continue
            if test == "advanced_vs_early_DN":
                yy = var[mk].astype(int)
                v, _ = hedges(R[g_ok].to_numpy()[mk][yy == 1], R[g_ok].to_numpy()[mk][yy == 0])
            else:
                v = np.array([stats.spearmanr(R[x].to_numpy()[mk], var[mk])[0] for x in g_ok])
            rows.append(pd.DataFrame({"cohort": name, "test": test, "gene": g_ok, "value": v, "n": int(mk.sum())}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    set_global_seed(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    P = RS["program"]
    core = load_core_table()
    tub = core[core.compartment == RS["compartment"]].set_index("gene")
    prog = pd.read_csv(OUT / "programs" / "programs.tsv.gz", sep="\t")
    prog = prog[prog.subset == "all"]
    sn = prog[prog.dataset == "snRNA"].set_index(["lineage", "state", "gene"])
    sc = prog[prog.dataset == "scRNA"].set_index(["lineage", "state", "gene"])
    rows = []
    for lin, sts in DRIVER_STATES.items():
        for st in sts:
            a = sn.xs((lin, st), level=(0, 1))
            b = sc.xs((lin, st), level=(0, 1)) if (lin, st) in sc.index.droplevel(2) else pd.DataFrame(columns=a.columns)
            for gname in tub.index.intersection(a.index):
                r = a.loc[gname]
                sgn = 1 if tub.at[gname, "direction"] == "up" else -1
                bl = b.log2fc.get(gname, np.nan)
                drv = (r.fdr < P["fdr"]) and (sgn * r.log2fc >= P["lfc"]) and np.isfinite(bl) and (sgn * bl >= P["rep_lfc"])
                rows.append({"gene": gname, "lineage": lin, "state": st, "scp_direction": tub.at[gname, "direction"],
                             "scp_g_re": tub.at[gname, "g_re"], "snRNA_log2fc": r.log2fc, "snRNA_fdr": r.fdr,
                             "snRNA_d_z": r.d_z, "snRNA_frac_donors_up": r.frac_donors_up, "scRNA_log2fc": bl,
                             "driver": bool(drv)})
    D = pd.DataFrame(rows)
    drivers = sorted(set(D.gene[D.driver]))
    genes = sorted(set(drivers) | set(WATCH))
    specs, means = [], []
    for name in RS["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        g_ok = [g for g in genes if g in set(pb.genes)]
        sp_, mn = state_specificity(pb, g_ok, donor_categories(pb, name))
        specs.append(sp_.assign(dataset=name)), means.append(mn.assign(dataset=name))
    SP = pd.concat(specs, ignore_index=True)
    ld = pd.read_csv(OUT / "concord" / "lineage_de.tsv.gz", sep="\t")
    ld = ld[(ld.dataset == "snRNA") & (ld.contrast == "CKD_vs_REF") & ld.gene.isin(genes)]
    ld = ld.pivot_table(index="gene", columns=["lineage", "lineage_cells"], values="log2fc")
    ld.columns = [f"kpmp_CKDvsREF_log2fc_{a}_{b}" for a, b in ld.columns]
    sh = pd.read_csv(OUT / "decomp" / "shapley_core_genes.tsv.gz", sep="\t")
    sh = sh[(sh.dataset == "snRNA") & (sh.contrast == "CKD_vs_REF")].set_index("gene")
    blocks = ["composition", "repair_state_fraction", "PT_TAL_normal_cell_profile", "PT_TAL_repair_cell_profile",
              "PT_TAL_other_altered_profile", "other_lineage_state_mix", "other_lineage_profile", "full"]
    sh = sh[blocks].add_prefix("tissue_delta_")
    IND = independent_effects(genes)
    iw = IND.pivot_table(index="gene", columns=["cohort", "test"], values="value")
    iw.columns = [f"indep_{a}_{b}" for a, b in iw.columns]
    dvc = IND[IND.test == "disease_vs_control"].pivot_table(index="gene", columns="cohort", values="value")
    meta = pd.read_csv(SCP_OUT / "core" / f"meta_{RS['compartment']}.tsv.gz", sep="\t").set_index("gene")
    sgn = np.sign(meta.g_re).reindex(dvc.index)                  # 方向 = bulk SCP meta 合并 g 的符号（核心外基因同样适用）
    snsp = SP[SP.dataset == "snRNA"].set_index("gene").drop(columns="dataset").add_prefix("snRNA_")
    scsp = SP[SP.dataset == "scRNA"].set_index("gene").drop(columns="dataset").add_prefix("scRNA_")
    Dg = D.pivot_table(index="gene", columns="state", values=["snRNA_log2fc", "scRNA_log2fc"])
    Dg.columns = [f"{a}_{b}" for a, b in Dg.columns]
    Dg = Dg.reindex(genes)
    Kt = pd.DataFrame(index=pd.Index(genes, name="gene"))
    Kt["in_scp_core_TUB"] = Kt.index.isin(tub.index)
    Kt["scp_direction"] = tub.direction.reindex(Kt.index)
    Kt["scp_g_re"] = tub.g_re.reindex(Kt.index)
    Kt["bulk_meta_g_re"] = meta.g_re.reindex(Kt.index)
    Kt["bulk_meta_fdr"] = meta.fdr.reindex(Kt.index)
    Kt["bulk_meta_sign_frac"] = meta.sign_frac.reindex(Kt.index)
    Kt["driver_of"] = [";".join(D.state[(D.gene == g) & D.driver]) for g in Kt.index]
    Kt["is_driver"] = Kt.driver_of != ""
    a_pt = sn.xs(("PT", "aPT"), level=(0, 1)).log2fc
    sgn = sgn.fillna(np.sign(a_pt.reindex(dvc.index)))           # 不在 bulk meta 宇宙中的基因（如 HAVCR1）：用 aPT 程序方向
    Kt["indep_sign_source"] = np.where(meta.g_re.reindex(Kt.index).notna(), "bulk_meta", "aPT_program")
    Kt["n_indep_cohorts_concordant"] = ((np.sign(dvc).mul(sgn, axis=0) > 0) & (dvc.abs() >= 0.5)).sum(1).reindex(Kt.index)
    Kt["n_indep_cohorts_tested"] = dvc.notna().sum(1).reindex(Kt.index)
    for ds_, tab in (("snRNA", sn), ("scRNA", sc)):              # 核心外的关注基因同样给出程序 log2FC / FDR
        for lin, sts in DRIVER_STATES.items():
            for st in sts:
                if (lin, st) in tab.index.droplevel(2):
                    t_ = tab.xs((lin, st), level=(0, 1))
                    Dg[f"{ds_}_log2fc_{st}"] = t_.log2fc.reindex(Kt.index).combine_first(
                        Dg.get(f"{ds_}_log2fc_{st}", pd.Series(dtype=float)))
                    Dg[f"{ds_}_fdr_{st}"] = t_.fdr.reindex(Kt.index)
    Kt = Kt.join(Dg).join(snsp).join(scsp).join(ld).join(sh).join(iw)
    if "tissue_delta_full" in Kt:
        Kt["tissue_share_repair_fraction"] = Kt.tissue_delta_repair_state_fraction / Kt.tissue_delta_full
        Kt["tissue_share_normal_cell_profile"] = Kt.tissue_delta_PT_TAL_normal_cell_profile / Kt.tissue_delta_full
        Kt["tissue_share_repair_cell_profile"] = Kt.tissue_delta_PT_TAL_repair_cell_profile / Kt.tissue_delta_full
        # 修复失败可归因的组织变化（Δlog2CPM）= 修复失败状态占比 + 修复失败细胞自身谱；用于排序关键基因（份额在小 Δ 时不稳定）
        Kt["tissue_delta_repair_attributable"] = (Kt.tissue_delta_repair_state_fraction
                                                  + Kt.tissue_delta_PT_TAL_repair_cell_profile)
        Kt["tissue_share_repair_attributable"] = Kt.tissue_delta_repair_attributable / Kt.tissue_delta_full
    Kt["rank_key"] = Kt.tissue_delta_repair_attributable.abs() if "tissue_delta_repair_attributable" in Kt else 0.0
    Kt = Kt.reset_index().sort_values(["is_driver", "rank_key"], ascending=[False, False]).drop(columns="rank_key")
    outs = {"driver_calls.tsv": D, "key_genes.tsv": Kt, "state_expression.tsv.gz": pd.concat(means, ignore_index=True),
            "independent_gene_effects.tsv.gz": IND}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/genes", [SCP_OUT / "core/core_genes.tsv", OUT / "programs/programs.tsv.gz",
                                               OUT / "concord/lineage_de.tsv.gz", OUT / "decomp/shapley_core_genes.tsv.gz"],
                     [O / f for f in outs], CFG["seed"], {"driver_states": DRIVER_STATES, "watch": WATCH})
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 20):
        print(D.groupby(["lineage", "state", "scp_direction"]).driver.agg(["sum", "size"]))
        cols = ["gene", "scp_direction", "scp_g_re", "bulk_meta_g_re", "driver_of", "snRNA_top_type",
                "snRNA_PT_repair_vs_PT_normal", "snRNA_TAL_repair_vs_TAL_normal", "tissue_delta_full",
                "tissue_share_repair_attributable", "tissue_share_normal_cell_profile", "n_indep_cohorts_concordant",
                "n_indep_cohorts_tested"]
        print(Kt[[c for c in cols if c in Kt]].round(3).head(120).to_string())


if __name__ == "__main__":
    main()
