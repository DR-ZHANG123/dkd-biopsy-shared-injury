"""Stage 18h：汇总——DKD 关键基因证据表（稳定性 + 各验证链）、8 病种图谱简表、集合层面验证汇总、总 PROVENANCE。

逐基因「通过」判据（每条链独立报告，不合成单一分数）：
  heldout   ：在 ≥1 个 ERCB 留出批次中被不含该批次的模型选中，且这些批次上的 Stouffer 单侧 P 经 BH（区室内关键基因）< 0.05；
  kpmp_ckd  ：snRNA 中该基因最高表达细胞类型内 DKD vs OTHER_CKD 与模型同向且 P < 0.05（供体为单位）；
  kpmp_hkd  ：同上，DKD vs HKD；
  egfr      ：DKD 供体全组织 pseudobulk 中与 eGFR 的 Spearman 方向为「越像 DKD eGFR 越低」且 P < 0.05。
性染色体基因（chrY、XIST）单独标记：DKD 与其他病种性别构成不同，这些基因的权重很可能反映性别而非病种生物学。
输出 results/18_interpret/summary/ 与 results/18_interpret/PROVENANCE.json。
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.kpmp_stats import bh  # noqa: E402
from lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
IC = CFG["interpret"]
R18 = ROOT / "results/18_interpret"
OUT = R18 / "summary"
EN = import_module("18_interpret_enrich")


def rd(sub: str, f: str) -> pd.DataFrame:
    return pd.read_csv(R18 / sub / f, sep="\t")


def sex_genes() -> set:
    lib = EN.read_lib(EN.fetch(IC["genesets"]["chrom_library"]))
    return set(lib.get("chrY", set())) | {"XIST", "TSIX"}


def dkd_evidence() -> pd.DataFrame:
    G = rd("weights", "key_genes.tsv")
    G = G[G["head"] == "DKD"][["compartment", "gene", "e_full", "w_prog", "w_resid", "prog_share", "topk_freq_e1",
                               "sign_consistency", "struct_full", "abs_rank"]]
    H = rd("validate", "heldout_key_genes_meta.tsv")
    H["fdr_heldout"] = np.nan
    for c, g in H.groupby("compartment"):
        ok = g.p_one_sided.notna()
        H.loc[g.index[ok], "fdr_heldout"] = bh(g.p_one_sided[ok].values)
    E = G.merge(H, on=["compartment", "gene"], how="left")
    loc = rd("kpmp", "localisation.tsv")
    for ds in IC["kpmp_datasets"]:
        l = loc[loc.dataset == ds].set_index("gene")[["top_type", "top_z", "tau", "top_log2cpm"]].add_prefix(f"{ds}_")
        E = E.merge(l, left_on="gene", right_index=True, how="left")
    D = rd("kpmp", "gene_direction.tsv")
    D = D[(D.role == "top") & D.set.str.startswith("key_")]
    for ds in IC["kpmp_datasets"]:
        for con in IC["kpmp_contrasts"]:
            d = D[(D.dataset == ds) & (D.contrast == con)][["compartment", "gene", "log2fc", "g", "p", "concordant"]]
            d = d.rename(columns={c: f"{ds}_{con}_{c}" for c in ("log2fc", "g", "p", "concordant")})
            E = E.merge(d, on=["compartment", "gene"], how="left")
    eg = rd("kpmp", "egfr_genes.tsv")
    eg = eg[(eg.donors == "DKD") & (eg.dataset == "snRNA")]
    eg["compartment"] = eg.set.str.replace("key_", "", regex=False)
    E = E.merge(eg[["compartment", "gene", "spearman_egfr", "p", "fdr"]].rename(
        columns={"p": "p_egfr", "fdr": "fdr_egfr"}), on=["compartment", "gene"], how="left")
    A = rd("enrich", "dkd_key_known_vs_novel.tsv")[["compartment", "gene", "annotation", "known_sources"]]
    E = E.merge(A, on=["compartment", "gene"], how="left")
    E["sex_chromosome_gene"] = E.gene.isin(sex_genes())
    s = np.sign(E.e_full)
    E["pass_heldout"] = E.fdr_heldout < 0.05
    E["pass_kpmp_ckd"] = (E["snRNA_DKD_vs_OTHER_CKD_concordant"] == True) & (E["snRNA_DKD_vs_OTHER_CKD_p"] < 0.05)  # noqa: E712
    E["pass_kpmp_hkd"] = (E["snRNA_DKD_vs_HKD_concordant"] == True) & (E["snRNA_DKD_vs_HKD_p"] < 0.05)  # noqa: E712
    E["pass_egfr"] = (np.sign(E.spearman_egfr) == -s) & (E.p_egfr < 0.05)
    E["n_chains_passed"] = E[["pass_heldout", "pass_kpmp_ckd", "pass_kpmp_hkd", "pass_egfr"]].sum(1)
    return E.sort_values(["compartment", "abs_rank"])


def atlas() -> tuple[pd.DataFrame, pd.DataFrame]:
    Q = rd("weights", "program_total_effects.tsv")
    G = rd("weights", "gene_effects.tsv.gz")
    S = rd("weights", "stability_summary.tsv")
    C = rd("programs", "program_concordance.tsv")
    sx = sex_genes()
    rows, genes = [], []
    for (comp, head), q in Q.groupby(["compartment", "head"]):
        k = q[q.is_key].sort_values("pi_full", key=np.abs, ascending=False)
        g = G[(G.compartment == comp) & (G["head"] == head) & G.is_key].sort_values("abs_rank")
        st = S[(S.compartment == comp) & (S["head"] == head)].set_index("level")
        cc = C[(C.compartment == comp) & (C["head"] == head) & (C.reference == "kpmp_snRNA")]
        rows.append({"compartment": comp, "head": head,
                     "key_programs": ", ".join(f"{r.program}{'+' if r.pi_full > 0 else '−'}" for r in k.itertuples()),
                     "top_up_genes": ", ".join(g[g.e_full > 0].gene.head(8)), "top_down_genes": ", ".join(g[g.e_full < 0].gene.head(8)),
                     "n_key_genes": len(g), "n_sex_chrom_key_genes": int(g.gene.isin(sx).sum()),
                     "gene_spearman_e1_median": st.at["gene", "spearman_e1_median"],
                     "gene_jaccard_top100_e1_median": st.at["gene", "jaccard_e1_median"],
                     "program_proj_spearman_e1_median": st.at["program_projection", "spearman_e1_median"],
                     "program_literal_spearman_e1_median": st.at["program", "spearman_e1_median"],
                     "spearman_pi_vs_stage13_kpmp_z": cc.spearman_pi_vs_z.iloc[0] if len(cc) else np.nan,
                     "sign_agree_pi_vs_z_absz2": cc.sign_agree_pi.iloc[0] if len(cc) else np.nan})
        genes.append(g.head(20)[["compartment", "head", "gene", "e_full", "topk_freq_e1", "sign_consistency", "prog_share"]]
                     .assign(sex_chromosome_gene=lambda d: d.gene.isin(sx)))
    return pd.DataFrame(rows), pd.concat(genes)


def set_level() -> pd.DataFrame:
    sd = rd("kpmp", "set_direction.tsv")
    sd = sd[sd.set.str.startswith("key_")].copy()
    sd["fdr"] = np.nan
    for _, g in sd.groupby(["dataset", "set", "contrast"]):
        ok = g.p_perm.notna()
        sd.loc[g.index[ok], "fdr"] = bh(g.p_perm[ok].values)
    out = sd.groupby(["dataset", "set", "contrast"]).agg(
        n_cell_types=("cell_type", "size"), n_delta_pos=("delta", lambda x: int((x > 0).sum())),
        n_p05=("p_perm", lambda x: int((x < .05).sum())), n_fdr05=("fdr", lambda x: int((x < .05).sum())),
        median_delta=("delta", "median")).reset_index()
    out["cell_types_fdr05"] = [", ".join(sd[(sd.dataset == r.dataset) & (sd.set == r.set) & (sd.contrast == r.contrast)
                                            & (sd.fdr < .05)].cell_type) for r in out.itertuples()]
    out["cell_types_p05"] = [", ".join(sd[(sd.dataset == r.dataset) & (sd.set == r.set) & (sd.contrast == r.contrast)
                                          & (sd.p_perm < .05)].cell_type) for r in out.itertuples()]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    E = dkd_evidence()
    A, AG = atlas()
    SL = set_level()
    files = {"dkd_key_gene_evidence.tsv": E, "atlas_heads.tsv": A, "atlas_top_genes.tsv": AG,
             "kpmp_set_direction_summary.tsv": SL}
    for f, t in files.items():
        t.to_csv(OUT / f, sep="\t", index=False)
    subs = sorted(R18.glob("*/PROVENANCE.json"))
    ins = [ROOT / "results/15_model/FROZEN_DESIGN.json"] + subs
    write_provenance("18_interpret", ins, [OUT / f for f in files], CFG["seed"],
                     {"substages": {p.parent.name: json.loads(p.read_text())["time"] for p in subs},
                      "scripts": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts/stages").glob("18_interpret*.py")),
                      "note": "RRG-ID frozen design retrained on all labelled patients (FULL) + every E1 fold; weights in models/18_interpret"})
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(A.round(2).to_string())
        print(SL.to_string())
        c = ["compartment", "gene", "e_full", "annotation", "sex_chromosome_gene", "snRNA_top_type", "stouffer_z",
             "fdr_heldout", "snRNA_DKD_vs_OTHER_CKD_log2fc", "snRNA_DKD_vs_OTHER_CKD_p", "snRNA_DKD_vs_HKD_p",
             "spearman_egfr", "p_egfr", "n_chains_passed"]
        print(E[c].round(3).to_string())


if __name__ == "__main__":
    main()
