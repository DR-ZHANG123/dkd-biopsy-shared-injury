"""Stage 24b：外部分数每方向基因数（top-k）的敏感性。

对 stage 11 的每个评估单元，用同一条外部损伤方向（results/11_injury/axis_<comp>_<unit>.tsv，未见该单元与 DKD）
在 k ∈ pathways_sensitivity.topk 下重算样本分数；报告 DKD vs 对照 AUROC，以及与主分析 k 的分数在病人中的 Spearman。
产出 results/24_pathways_sensitivity/topk/topk_auroc.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import full_ranks  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "24_pathways_sensitivity" / "topk"
K0 = CFG["injury"]["top_k"]


def score(R: pd.DataFrame, axis: pd.Series, k: int) -> pd.Series:
    ax = axis.reindex(R.columns).dropna()
    return R[ax.nlargest(k).index].mean(axis=1) - R[ax.nsmallest(k).index].mean(axis=1)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(ROOT / "data/processed/samples.tsv", sep="\t").set_index("sample_uid")
    sc = pd.read_csv(ROOT / "results/11_injury/injury_scores.tsv", sep="\t")
    rows, ins = [], []
    for (unit, comp), g in sc.groupby(["eval_unit", "compartment"]):
        p = ROOT / f"results/11_injury/axis_{comp}_{unit}.tsv"
        ins.append(p)
        axis = pd.read_csv(p, sep="\t").set_index("gene").injury_effect
        R = full_ranks(pd.Index(g.sample_uid), samples).dropna(axis=1)
        diag = g.set_index("sample_uid").diagnosis.reindex(R.index)
        ref = score(R, axis, K0)
        pat = diag != "CONTROL"
        dc = diag.isin(["DKD", "CONTROL"])
        for k in CFG["pathways_sensitivity"]["topk"]:
            s = score(R, axis, k)
            rows.append({"eval_unit": unit, "compartment": comp, "k": k, "n_dkd": int((diag == "DKD").sum()),
                         "n_control": int((diag == "CONTROL").sum()),
                         "auroc_dkd_vs_control": roc_auc_score((diag[dc] == "DKD").astype(int), s[dc]),
                         "spearman_with_main_k_patients": spearmanr(s[pat], ref[pat])[0]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "topk_auroc.tsv", sep="\t", index=False)
    write_provenance("24_pathways_sensitivity/topk", ins + [ROOT / "results/11_injury/injury_scores.tsv"],
                     [OUT / "topk_auroc.tsv"], CFG["seed"], {"main_k": K0})
    print(t.pivot(index="eval_unit", columns="k", values="auroc_dkd_vs_control").round(3).to_string())
    print(t.pivot(index="eval_unit", columns="k", values="spearman_with_main_k_patients").round(3).to_string())


if __name__ == "__main__":
    main()
