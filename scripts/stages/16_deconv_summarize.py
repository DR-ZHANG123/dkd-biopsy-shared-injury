"""Stage 16e：汇总反卷积可行性研究的三个结论表到 results/16_deconv/（只读下游表，不重新计算模型）。

feature_set_comparison.tsv  各参照 × 特征集：dev 非 DKD 任务（E1/E2/E4 合并、ERCB 子集）原始 / 调整 AUROC，
                            相对 B-L2 的 cell 级配对差（bootstrap 95% CI、胜率、Wilcoxon p）。
plausibility_key.tsv        各参照 × 方法（BayesPrism / NNLS）× 区室的关键比例（批中位数的中位数、批间范围）。
axis_r2_summary.tsv         共享轴分数被组成解释的 CV R²（单元中位数与范围），并列同维度 bulk 主成分与随机基因集对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
DC = CFG["deconv"]
RES = ROOT / "results" / "16_deconv"
TAG = DC["eval"]["run_tag"]
COLS = ["n_cells", "auroc_ref", "adj_auroc_ref", "auroc_alt", "adj_auroc_alt", "d_auroc_mean", "d_auroc_lo",
        "d_auroc_hi", "d_auroc_frac_ref_better", "d_auroc_wilcoxon_p", "d_adj_auroc_mean", "d_adj_auroc_lo",
        "d_adj_auroc_hi", "d_adj_auroc_frac_ref_better", "d_adj_auroc_wilcoxon_p"]


def comparison() -> pd.DataFrame:
    out = []
    for ref in DC["references"]:
        f = RES / "eval" / ref / "vs_BL2.tsv"
        if f.exists():
            v = pd.read_csv(f, sep="\t")
            v = v[v.family.isin(["ALL", "ERCB", "E1", "E2", "E4"])]
            out.append(v[["method", "baseline", "family"] + COLS].assign(reference=ref))
    t = pd.concat(out)
    return t.rename(columns=lambda c: c.replace("_ref", "_method").replace("_alt", "_BL2")
                    if c.endswith(("_ref", "_alt")) else c.replace("frac_ref_better", "win_rate"))


def plausibility() -> pd.DataFrame:
    k = pd.read_csv(RES / "plausibility" / TAG / "key_checks.tsv", sep="\t")
    k = k[k.subset == "all"]
    return (k.groupby(["reference", "method", "compartment", "quantity"])["median"]
            .agg(median_of_batches="median", min_batch="min", max_batch="max", n_batches="size").reset_index())


def axis_r2() -> pd.DataFrame:
    a = pd.read_csv(RES / "plausibility" / TAG / "axis_r2.tsv", sep="\t")
    a["features"] = a.features.str.replace(r"^bulkPC\d+$", "bulkPC(same_p)", regex=True) \
                              .str.replace(r"^random\d+x\d+$", "random_sets(same_p)", regex=True)
    a = a[a.features != "COMP+SF"]
    return (a.groupby(["reference", "compartment", "features"])
            .agg(n_units=("unit", "size"), r2_median=("r2", "median"), r2_cv_median=("r2_cv", "median"),
                 r2_cv_min=("r2_cv", "min"), r2_cv_max=("r2_cv", "max")).reset_index())


def main() -> None:
    outs = {"feature_set_comparison.tsv": comparison(), "plausibility_key.tsv": plausibility(),
            "axis_r2_summary.tsv": axis_r2()}
    for f, df in outs.items():
        df.to_csv(RES / f, sep="\t", index=False)
    pd.set_option("display.width", 250)
    c = outs["feature_set_comparison.tsv"]
    print(c[c.family.isin(["ALL", "ERCB"])][["reference", "method", "family", "n_cells", "auroc_method",
                                             "adj_auroc_method", "d_adj_auroc_mean", "d_adj_auroc_lo",
                                             "d_adj_auroc_hi", "d_adj_auroc_win_rate",
                                             "d_adj_auroc_wilcoxon_p"]].round(3).to_string(index=False))
    print(outs["axis_r2_summary.tsv"].round(3).to_string(index=False))
    ins = [RES / "plausibility" / TAG / f for f in ("key_checks.tsv", "axis_r2.tsv")]
    ins += [RES / "eval" / r / "vs_BL2.tsv" for r in DC["references"]]
    write_provenance("16_deconv", ins, [RES / f for f in outs], CFG["seed"],
                     {"primary_reference": DC["primary"], "run_tag": TAG,
                      "stages": ["16_deconv_prepare.py", "16_deconv_run.py (+16_deconv_bayesprism.R)",
                                 "16_deconv_plausibility.py", "16_deconv_eval.py", "16_deconv_summarize.py"]})


if __name__ == "__main__":
    main()
