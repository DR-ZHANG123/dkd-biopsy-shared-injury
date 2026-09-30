"""Stage 18f：细胞类型程序图谱与 stage 13 / 14 结论的对照（一致 / 不一致都报告）。

对每个 (区室, head, 程序)：
  模型侧：pi_full / z_enrich_full（程序总效应投影与标志富集，lib.interp.program_projection；主量）、
          b_full（字面程序通路偏效应，冻结模型中 ≈ 0）与 struct_full（结构系数 = 程序分数与集成 head 分数在训练病人中
          stratum 内中心化后的 Pearson r，边际关联；程序间共线时二者可异号）、稳定性、is_key；
  bulk 富集：stage 13 <head>_vs_PAT_adj（META 与各单元）的细胞类型富集 z——旧标志（stage 13）与 KPMP 标志
          （stage 14 snRNA_endosplit / scRNA_endosplit，与本模型程序同名）；注意这些富集与模型用同一 bulk 队列，
          属内部一致性而非独立验证；
  KPMP 丰度（仅 DKD，独立）：stage 14 abundance_tests 中 DKD vs REF / HKD / OTHER_CKD 的供体水平比例 Hedges g 与置换 P。
汇总：每 (区室, head) 的 Spearman(pi, z)、Spearman(struct, z)，以及 |z| ≥ 2 的类型中符号一致率。
输出 results/18_interpret/programs/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
W = ROOT / "results/18_interpret/weights"
OUT = ROOT / "results/18_interpret/programs"
S13, S14 = ROOT / "results/13_celltype", ROOT / "results/14_kpmp"
REFS = {"stage13_old": None, "kpmp_snRNA": "snRNA_endosplit", "kpmp_scRNA": "scRNA_endosplit"}


def enrichment_long() -> pd.DataFrame:
    old = pd.read_csv(S13 / "celltype_enrichment.tsv", sep="\t").assign(reference="stage13_old")
    new = pd.read_csv(S14 / "celltype_enrichment_kpmp.tsv", sep="\t")
    new = new[new.reference.isin([v for v in REFS.values() if v])]
    new["reference"] = new.reference.map({v: k for k, v in REFS.items() if v})
    e = pd.concat([old, new], ignore_index=True)
    e = e[e.contrast.str.endswith("_vs_PAT_adj")].copy()
    e["head"] = e.contrast.str.replace("_vs_PAT_adj", "", regex=False)
    return e.rename(columns={"cell_type": "program"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(W / "program_total_effects.tsv", sep="\t").merge(
        pd.read_csv(W / "program_weights.tsv", sep="\t")[["compartment", "head", "program", "b_full", "struct_full"]],
        on=["compartment", "head", "program"])
    E = enrichment_long()
    wide = E.pivot_table(index=["compartment", "head", "program"], columns=["reference", "unit"], values="z")
    wide.columns = [f"z_{r}_{u}" for r, u in wide.columns]
    T = P.merge(wide.reset_index(), on=["compartment", "head", "program"], how="left")
    ab = pd.read_csv(S14 / "abundance_tests.tsv", sep="\t")
    ab = ab[(ab.denominator == "all") & ab.contrast.isin(["DKD_vs_REF", "DKD_vs_HKD", "DKD_vs_OTHER_CKD"])]
    abw = ab.pivot_table(index="cell_type", columns=["dataset", "contrast"], values=["hedges_g", "perm_p"])
    abw.columns = [f"abund_{d}_{c}_{v}" for v, d, c in abw.columns]
    abw = abw.reset_index().rename(columns={"cell_type": "program"}).assign(head="DKD")
    T = T.merge(abw, on=["head", "program"], how="left")
    rows = []
    for (comp, head), g in T.groupby(["compartment", "head"]):
        for ref in REFS:
            col = f"z_{ref}_META"
            if col not in g or g[col].notna().sum() < 5:
                continue
            x = g.dropna(subset=[col])
            sig = x[x[col].abs() >= 2]
            rows.append({"compartment": comp, "head": head, "reference": ref, "n_programs": len(x),
                         "spearman_pi_vs_z": spearmanr(x.pi_full, x[col])[0],
                         "spearman_struct_vs_z": spearmanr(x.struct_full, x[col])[0],
                         "n_abs_z_ge2": len(sig),
                         "sign_agree_pi": float((np.sign(sig.pi_full) == np.sign(sig[col])).mean()) if len(sig) else np.nan,
                         "sign_agree_struct": float((np.sign(sig.struct_full) == np.sign(sig[col])).mean()) if len(sig) else np.nan,
                         "key_programs": ",".join(f"{r.program}{'+' if r.pi_full > 0 else '-'}" for r in x[x.is_key].itertuples()),
                         "key_vs_z": ";".join(f"{r.program}:{'agree' if np.sign(r.pi_full) == np.sign(getattr(r, col)) else 'disagree'}"
                                              f"(z={getattr(r, col):.1f})" for r in x[x.is_key].itertuples())})
    S = pd.DataFrame(rows)
    T.to_csv(OUT / "program_atlas.tsv", sep="\t", index=False)
    S.to_csv(OUT / "program_concordance.tsv", sep="\t", index=False)
    write_provenance("18_interpret/programs", [W / "program_weights.tsv", S13 / "celltype_enrichment.tsv",
                                               S14 / "celltype_enrichment_kpmp.tsv", S14 / "abundance_tests.tsv"],
                     [OUT / "program_atlas.tsv", OUT / "program_concordance.tsv"], CFG["seed"])
    with pd.option_context("display.width", 250, "display.max_colwidth", 200):
        print(S.round(2).to_string())
        c = ["compartment", "program", "pi_full", "z_enrich_full", "struct_full", "sign_consistency", "is_key", "z_stage13_old_META",
             "z_kpmp_snRNA_META", "abund_snRNA_DKD_vs_OTHER_CKD_hedges_g", "abund_snRNA_DKD_vs_OTHER_CKD_perm_p"]
        print(T[T["head"] == "DKD"].sort_values(["compartment", "abs_rank"])[[x for x in c if x in T]].round(3).to_string())


if __name__ == "__main__":
    main()
