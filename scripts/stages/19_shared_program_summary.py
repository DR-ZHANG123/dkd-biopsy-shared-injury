"""Stage 19f：主图 tidy 表（results/19_shared_program/figdata/）与关键数字；写 stage 19 顶层 PROVENANCE.json。
只读 19a–19e 的结果表（及 KPMP 供体元数据），不重新计算任何模型。
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp_core import OUT, SP  # noqa: E402

CFG = load_config()
F = OUT / "figdata"
EGFR = "Baseline eGFR (ml/min/1.73m2) (Binned)"


def rd(sub: str, name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / sub / name, sep="\t")


def core_panels() -> dict[str, pd.DataFrame]:
    core = rd("core", "core_genes.tsv")
    n = SP["n_representative"]
    rep = (core.assign(a=core.g_re.abs()).sort_values("a", ascending=False)
           .groupby(["compartment", "direction"]).head(n).drop(columns="a"))
    vol = []
    for comp in SP["sources"]:
        m = rd("core", f"meta_{comp}.tsv.gz")
        m["representative"] = m.gene.isin(rep.gene[rep.compartment == comp])
        vol.append(m[["compartment", "gene", "g_re", "se", "fdr", "I2", "sign_frac", "is_core", "direction", "representative"]])
    src = [c for c in rep.columns if c.startswith("g_") and c not in ("g_re",)]
    forest = rep.melt(id_vars=["compartment", "gene", "direction", "g_re", "se", "I2"], value_vars=src,
                      var_name="source", value_name="g_source").dropna(subset=["g_source"])
    forest["source"] = forest.source.str.replace("^g_", "", regex=True)
    return {"fig1_core_volcano": pd.concat(vol), "fig1_core_forest": forest,
            "fig1_disease_concordance": rd("core", "disease_concordance.tsv")}


def cell_panels() -> dict[str, pd.DataFrame]:
    loc = rd("cells", "core_top_type_enrichment.tsv")
    comp = rd("cells", "composition_r2.tsv")
    dt = rd("kpmp", "decomp_tests.tsv")
    dt = dt[dt.contrast.isin(["CKD_vs_REF", "CKD_perc_vs_REF_perc"])].copy()
    dt["kind"] = np.where(dt.scenario.isin(["full", "comp_only", "state_only"]), dt.scenario,
                          dt.scenario.str.split("_").str[0] + "_by_type")
    dt["cell_type"] = np.where(dt.kind.str.endswith("_by_type"), dt.scenario.str.split("_", n=1).str[1], "")
    gc = rd("kpmp", "gene_class.tsv")
    gcs = gc.groupby(["dataset", "contrast", "compartment", "direction", "home_type", "class"]).size().rename("n_genes").reset_index()
    ab = rd("kpmp", "abundance.tsv")
    return {"fig2_core_localisation": loc, "fig2_marker_program_z": rd("cells", "marker_program_z.tsv"),
            "fig2_composition_r2": comp, "fig2_kpmp_decomposition": dt, "fig2_kpmp_gene_class": gcs,
            "fig2_kpmp_abundance": ab[ab.contrast.isin(["CKD_vs_REF", "CKD_perc_vs_REF_perc"])],
            "fig2_kpmp_set_direction": rd("kpmp", "set_direction.tsv")}


def kpmp_egfr_scatter() -> pd.DataFrame:
    mid = import_module("18_interpret_kpmp").egfr_mid
    sc = pd.read_csv(OUT / "kpmp" / "decomp_scores.tsv.gz", sep="\t")
    sc = sc[(sc.dataset == "snRNA") & sc.scenario.isin(["full", "comp_only", "state_only"])]
    d = pd.read_csv(ROOT / CFG["kpmp"]["pseudobulk_cache"] / "snRNA" / "donors.tsv", sep="\t")
    d["donor"] = d.donor.astype(str)
    d["egfr_mid"] = d[EGFR].map(mid)
    d["group"] = np.select([d.DKD.astype(str) == "True", d.OTHER_CKD.astype(str) == "True", d.REF.astype(str) == "True"],
                           ["DKD", "OTHER_CKD", "REF"], "other")
    sc["donor"] = sc.donor.astype(str)
    return sc.merge(d[["donor", "group", "egfr_mid", "TissueCollection"]], on="donor")


def replication_panels() -> dict[str, pd.DataFrame]:
    a = rd("replicate", "auroc.tsv")
    c = rd("replicate", "clinical.tsv")
    e = rd("kpmp", "egfr.tsv")
    e = e.assign(cohort="KPMP_snRNA_donors", test="egfr_bin_" + e.donors, core=e.compartment, gene_set="full",
                 primary=e.compartment == "TUB")
    e = e[e.scenario == "full"].rename(columns={"pct_more_negative_than_random": "pct_more_extreme_than_random"})
    c["pct_more_extreme_than_random"] = np.where(c.test.str.startswith("egfr"), 1 - c.pct_vs_random, c.pct_vs_random)
    cl = pd.concat([c, e[[x for x in e.columns if x in c.columns or x == "pct_more_extreme_than_random"]]])
    return {"fig3_replication_auroc": a, "fig3_clinical": cl, "fig3_kpmp_egfr_scatter": kpmp_egfr_scatter(),
            "fig3_scores": rd("replicate", "scores.tsv")}


def residual_panels() -> dict[str, pd.DataFrame]:
    return {"fig4_residual_programs": rd("residual", "cell_programs.tsv"),
            "fig4_residual_top_genes": rd("residual", "top_genes.tsv"),
            "fig4_residual_tests": rd("residual", "independent_tests.tsv"),
            "fig4_disease_scp_shift": rd("residual", "disease_scp_shift.tsv")}


def key_numbers(T: dict[str, pd.DataFrame]) -> pd.DataFrame:
    core = rd("core", "core_genes.tsv")
    rows = [{"item": f"n_core_{c}_{d}", "value": int(n)} for (c, d), n in core.groupby(["compartment", "direction"]).size().items()]
    rows.append({"item": "n_core_shared_both_compartments", "value": int(core[core.compartment == "TUB"].core_in_other_compartment.sum())})
    r2 = T["fig2_composition_r2"]
    for (ref, comp), g in r2.groupby(["reference", "compartment"]):
        rows.append({"item": f"composition_r2cv_median_{ref}_{comp}", "value": float(g.r2_cv.median())})
    dt = T["fig2_kpmp_decomposition"]
    for _, r in dt[dt.kind.isin(["full", "comp_only", "state_only"])].iterrows():
        rows.append({"item": f"kpmp_{r.dataset}_{r.compartment}_{r.contrast}_{r.scenario}_auroc", "value": r.auroc})
    a = T["fig3_replication_auroc"]
    for _, r in a[a.primary & (a.gene_set == "full")].iterrows():
        rows.append({"item": f"replication_{r.cohort}_auroc", "value": r.auroc})
        rows.append({"item": f"replication_{r.cohort}_pct_vs_random", "value": r.pct_vs_random})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    F.mkdir(parents=True, exist_ok=True)
    T = core_panels() | cell_panels() | replication_panels() | residual_panels()
    outs = []
    for k, t in T.items():
        f = F / (f"{k}.tsv.gz" if len(t) > 20000 else f"{k}.tsv")
        t.to_csv(f, sep="\t", index=False)
        outs.append(f)
    kn = key_numbers(T)
    kn.to_csv(F / "key_numbers.tsv", sep="\t", index=False)
    outs.append(F / "key_numbers.tsv")
    subs = [OUT / s / "PROVENANCE.json" for s in ("core", "cells", "kpmp", "replicate", "residual")]
    write_provenance("19_shared_program", subs, outs, CFG["seed"],
                     {"substages": ["core", "cells", "kpmp", "replicate", "residual"], "config_section": "shared_program"})
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(kn.round(3).to_string())


if __name__ == "__main__":
    main()
