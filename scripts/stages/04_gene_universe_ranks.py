"""Stage 04：基因宇宙、样本内百分位秩、样本表与 LOCO folds（接口见 notes/02_interfaces.md）。

基因宇宙：所有评估 / 外部测试 cohort 实际用到的 series 版本都测到的基因（覆盖率 ≥ min_platform_coverage），
再按预训练语料（keep_for_pretrain）中的 series 内合并秩方差取 top n_nodes。
用 series 内方差而非全体方差：跨平台的探针效应会让全体秩方差偏向「平台差异大」的基因。
秩：在该样本全部测到的基因上排序（平均秩处理 ties），(rank − 1)/(n − 1)，再取宇宙子集；未测到 = NaN。
产出：data/processed/{genes.txt, ranks.parquet, samples.tsv, folds.tsv, cohorts.tsv}
      results/04_ranks/{cohort_table.tsv, fold_summary.tsv, gene_universe.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cohorts import assign_cohorts, cohort_counts, make_folds  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
CC = CFG["cohorts"]
INTERIM = ROOT / CFG["paths"]["interim"]
PROC = ROOT / CFG["paths"]["processed"]
OUT = ROOT / "results" / "04_ranks"
SAMPLE_COLS = ["sample_uid", "series", "gse", "gsm", "technology", "compartment", "role", "diagnosis",
               "dup_group", "keep_for_pretrain"]


def load_expr() -> dict[str, pd.DataFrame]:
    return {p.name.replace("_expr.parquet", ""): pd.read_parquet(p) for p in sorted(INTERIM.glob("*_expr.parquet"))}


def percentile_ranks(expr: pd.DataFrame) -> pd.DataFrame:
    """每个样本在自身全部非缺失基因上排序：(rank − 1)/(n − 1)，平均秩处理 ties。"""
    out = np.full(expr.shape, np.nan, dtype=np.float64)
    v = expr.values
    for j in range(v.shape[1]):
        ok = ~np.isnan(v[:, j])
        r = rankdata(v[ok, j], method="average")
        out[ok, j] = (r - 1) / (ok.sum() - 1)
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


def build_samples(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lab = pd.read_csv(ROOT / "results" / "03_labels" / "sample_labels.tsv", sep="\t", dtype={"gsm": str})
    st = pd.read_csv(ROOT / "results" / "02_overlap" / "sample_table.tsv", sep="\t", usecols=["sample_uid", "dup_group"])
    s = lab.merge(st, on="sample_uid", how="left", validate="1:1").set_index("sample_uid")
    assert s.dup_group.notna().all()
    s["technology"] = np.where(s.series.str.endswith("RNAseq"), "rnaseq", "array")
    s["technical_outlier"] = s.technical_outlier.astype(str).str.lower().eq("true")
    n_samples = sum(v.shape[1] for v in data.values())
    assert len(s) == n_samples, f"标签表 {len(s)} 行 ≠ 表达矩阵 {n_samples} 列"
    return s


def keep_for_pretrain(s: pd.DataFrame, cohort: pd.Series, n_genes: dict[str, int]) -> pd.Series:
    """非外部测试样本；同一 dup_group 只保留一个代表：评估 cohort 代表样本 > 基因数多的版本 > series 名 > GSM。
    与外部测试样本同 dup_group 的样本一律不保留。"""
    ext = s.role.str.startswith("external")
    ext_groups = set(s.loc[ext, "dup_group"])
    cand = s[~ext & ~s.dup_group.isin(ext_groups)].copy()
    cand["is_rep"] = cand.index.isin(cohort.index)
    cand["ng"] = cand.series.map(n_genes)
    cand = cand.sort_values(["dup_group", "is_rep", "ng", "series", "gsm"], ascending=[True, False, False, True, True])
    keep = cand.drop_duplicates("dup_group").index
    return pd.Series(s.index.isin(keep), index=s.index)


def gene_universe(data: dict[str, pd.DataFrame], eval_series: list[str]) -> tuple[pd.Index, pd.DataFrame]:
    measured = {k: data[k].index[data[k].notna().all(axis=1)] for k in eval_series}
    allg = pd.Index(sorted(set().union(*measured.values())))
    cov = pd.DataFrame({k: allg.isin(v) for k, v in measured.items()}, index=allg)
    frac = cov.mean(axis=1)
    return frac.index[frac >= CFG["genes"]["min_platform_coverage"]], cov


def pooled_within_series_var(ranks: dict[str, pd.DataFrame], s: pd.DataFrame, genes: pd.Index) -> pd.Series:
    """预训练语料中逐 series 的秩方差按 (n − 1) 加权合并；某 series 未测到的基因不计入该 series。"""
    num = pd.Series(0.0, index=genes)
    den = pd.Series(0.0, index=genes)
    for k, r in ranks.items():
        cols = [g for g in r.columns if s.at[f"{k}|{g}", "keep_for_pretrain"]]
        if len(cols) < 3:
            continue
        x = r.reindex(genes)[cols]
        ok = x.notna().all(axis=1)
        num[ok] += x[ok].var(axis=1) * (len(cols) - 1)
        den[ok] += len(cols) - 1
    return (num / den.replace(0, np.nan)).rename("pooled_within_series_var")


def main() -> None:
    set_global_seed(CFG["seed"])
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_expr()
    n_genes = {k: int(v.notna().all(axis=1).sum()) for k, v in data.items()}
    s = build_samples(data)

    cohort = assign_cohorts(s, CC, n_genes)
    table = cohort_counts(s, cohort, CC)
    table.to_csv(OUT / "cohort_table.tsv", sep="\t", index=False)
    s["keep_for_pretrain"] = keep_for_pretrain(s, cohort, n_genes)

    eval_series = sorted(s.loc[cohort.index, "series"].unique())
    universe, cov = gene_universe(data, eval_series)
    ranks = {k: percentile_ranks(v) for k, v in data.items()}
    var = pooled_within_series_var(ranks, s, universe)
    genes = var.sort_values(ascending=False, kind="mergesort").index[:CFG["genes"]["n_nodes"]]
    genes = pd.Index(sorted(genes))
    assert len(genes) == CFG["genes"]["n_nodes"], f"宇宙只有 {len(universe)} 个基因"
    (PROC / "genes.txt").write_text("\n".join(genes) + "\n")
    pd.DataFrame({"gene": universe, "pooled_within_series_var": var.reindex(universe).values,
                  "selected": universe.isin(genes)}).to_csv(OUT / "gene_universe.tsv", sep="\t", index=False)
    cov.sum().rename("n_genes_measured").rename_axis("series").to_csv(OUT / "eval_platform_coverage.tsv", sep="\t")

    rk = pd.concat([ranks[k].reindex(genes).T.set_axis([f"{k}|{c}" for c in ranks[k].columns]) for k in ranks])
    rk = rk.loc[s.index].astype("float32")
    rk.index.name = "sample_uid"
    rk.to_parquet(PROC / "ranks.parquet")

    s = s.reset_index().sort_values(["series", "gsm"])
    s[SAMPLE_COLS].to_csv(PROC / "samples.tsv", sep="\t", index=False)
    cdf = cohort.rename("cohort").rename_axis("sample_uid").reset_index()
    # ERCB cohort 可能混有不同 CDF 版本（GPL）；标出主 CDF 版本样本，供下游做「仅主版本」敏感性分析
    cdf["series"] = s.set_index("sample_uid").loc[cdf.sample_uid, "series"].values
    gpl = cdf.series.str.split("-").str[1]
    cdf["majority_version"] = gpl == gpl.groupby(cdf.cohort).transform(lambda x: x.mode().iat[0])
    cdf.to_csv(PROC / "cohorts.tsv", sep="\t", index=False)
    s = s.set_index("sample_uid")
    folds = make_folds(s, cohort, table, CC)
    folds.to_csv(PROC / "folds.tsv", sep="\t", index=False)
    check_folds(folds, s)
    fsum = folds.merge(s[["diagnosis"]], left_on="sample_uid", right_index=True).assign(
        pos=lambda d: d.diagnosis.eq("DKD")).groupby(["task", "compartment", "fold", "test_cohort", "split"]).agg(
        n=("pos", "size"), n_DKD=("pos", "sum")).reset_index()
    fsum.to_csv(OUT / "fold_summary.tsv", sep="\t", index=False)

    outs = [PROC / f for f in ("genes.txt", "ranks.parquet", "samples.tsv", "folds.tsv", "cohorts.tsv")]
    outs += [OUT / f for f in ("cohort_table.tsv", "fold_summary.tsv", "gene_universe.tsv", "eval_platform_coverage.tsv")]
    write_provenance("04_ranks", sorted(INTERIM.glob("*_expr.parquet")) + [
        ROOT / "results" / "03_labels" / "sample_labels.tsv", ROOT / "results" / "02_overlap" / "sample_table.tsv"],
        outs, CFG["seed"],
        {"n_samples": int(len(s)), "n_dup_groups": int(s.dup_group.nunique()),
         "n_keep_for_pretrain": int(s.keep_for_pretrain.sum()), "n_universe": int(len(universe)),
         "n_nodes": int(len(genes)), "eval_series": eval_series,
         "nan_frac_ranks": float(np.isnan(rk.values).mean())})
    print(table.drop(columns=["series_versions", "other_CKD_breakdown", "excluded_dx_breakdown"]).to_string())
    print(fsum.to_string())
    print("universe", len(universe), "nodes", len(genes), "keep_for_pretrain", int(s.keep_for_pretrain.sum()))


def check_folds(folds: pd.DataFrame, s: pd.DataFrame) -> None:
    """每个 fold：dup_group 不跨 train/test；每个 cohort 内一个 dup_group 只出现一次。"""
    for key, f in folds.groupby(["task", "compartment", "fold"]):
        g = s.loc[f.sample_uid, "dup_group"].values
        tr = set(g[(f.split == "train").values])
        te = set(g[(f.split == "test").values])
        assert not tr & te, f"{key}: dup_group 跨 train/test"
        assert len(g) == len(set(g)), f"{key}: fold 内同一 dup_group 出现多次"


if __name__ == "__main__":
    main()
