"""Stage 16d：反卷积特征集在 stage 15 dev 非 DKD 任务上的比较（同 fold、同训练病人、同 L2 logistic、同评估口径）。

特征集（config deconv.eval.methods）：RANK = stage 15 的组织平均样本内秩（B-L2 原样复现，含 top-方差基因筛选）；
COMP = 类型层组成（CLR）；SF = 类型内状态份额（logit）；Z = 细胞类型特异表达的样本内秩（组成之外的细胞内状态）。
多块组合时，每块先按训练病人标准化，再把各块总方差调到与最大块相同（避免 10 维组成被数千维表达淹没）。
DKD 样本不进入任何测试集（assert_no_dkd_eval）；DKD vs 其他 CKD 的最终评估不在此 stage。
输出 results/16_deconv/eval/<参照>/：preds/<method>/<fold>.tsv、dev_metrics.tsv、dev_summary.tsv、vs_BL2.tsv。
用法：python scripts/stages/16_deconv_eval.py [--reference snRNA] [--families E1,E2,E4] [--methods all] [--n-jobs 48]
"""
from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.deconv_feats import feature_sets  # noqa: E402
from lib.evalkit import fit_logistic_cv  # noqa: E402
from lib.m2_baselines import _top_var, run_baseline  # noqa: E402
from lib.m2_data import M2, covariate, features, impute, make_folds  # noqa: E402
from lib.m2_eval import assert_no_dkd_eval, metrics_from_preds  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
EV = CFG["deconv"]["eval"]
DEV = import_module("15_dev")
SUM = import_module("15_summarize")
_CACHE: dict = {}


def blocks_for(comp: str, ref: str) -> dict[str, pd.DataFrame]:
    key = (comp, ref)
    if key not in _CACHE:
        _CACHE[key] = feature_sets(comp, EV["run_tag"], ref, features())
    return _CACHE[key]


def _unit_center(X: np.ndarray, units: np.ndarray) -> np.ndarray:
    out = X.copy()
    for u in np.unique(units):
        m = units == u
        out[m] -= X[m].mean(0)
    return out


def _block_arrays(fold, B: pd.DataFrame, pat: np.ndarray, cap: bool, uc: bool):
    """uc：每个单元（训练与测试单元各自）用该单元全部样本（病人 + 对照）的均值中心化 —— 无标签的批次校正。"""
    Xtr, Xte, _ = impute(B.reindex(fold.train.index).to_numpy(np.float32),
                         B.reindex(fold.test.index).to_numpy(np.float32))
    if uc:
        Xtr, Xte = _unit_center(Xtr, fold.train.unit.to_numpy()), _unit_center(Xte, fold.test.unit.to_numpy())
    if cap:
        g = _top_var(Xtr[pat], M2["baseline_max_genes"])
        Xtr, Xte = Xtr[:, g], Xte[:, g]
    mu, sd = Xtr[pat].mean(0), Xtr[pat].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def design(fold, names: list[str], comp: str, ref: str):
    """按块构建训练 / 测试矩阵；高维块（RANK*、Z、RESID）与 B-L2 同法取训练病人方差 top 基因。
    块名后缀「|uc」= 单元中心化。"""
    bl = blocks_for(comp, ref)
    pat = fold.train.role.isin(["head", "neg"]).to_numpy()
    parts = []
    for n in names:
        base, _, mod = n.partition("|")
        parts.append(_block_arrays(fold, bl[base], pat, base in ("RANK", "RANK_Z", "Z", "RESID"), mod == "uc"))
    if len(parts) > 1:
        pmax = max(p[0].shape[1] for p in parts)
        parts = [(a * np.sqrt(pmax / a.shape[1]), b * np.sqrt(pmax / a.shape[1])) for a, b in parts]
    return np.hstack([p[0] for p in parts]), np.hstack([p[1] for p in parts]), pat


def fit_scores(Xtr, Xte, train, pat, d, seed) -> np.ndarray:
    y = (train.diagnosis.to_numpy() == d).astype(int)
    cfg = dict(CFG)
    cfg["baselines"] = dict(CFG["baselines"], standardize=False)   # 已在 design() 中按训练病人标准化
    f = fit_logistic_cv(Xtr[pat], y[pat], train.unit.to_numpy()[pat], CFG["probe"]["C_grid"], "l2", cfg, seed)
    return f["model"].decision_function(Xte)


def run_job(method: str, fam: str, comp: str, i: int, ref: str, path: str) -> str:
    t0 = time.time()
    folds = make_folds(fam, comp)
    fold = folds[i]
    cells = DEV.fold_cells(folds, fam)[fold.key]
    assert_no_dkd_eval(fold.test)
    a = covariate(fold)
    names = EV["methods"][method]
    scores = {}
    if names == ["RANK"]:                       # B-L2：与 stage 15 完全相同的代码路径
        F = features()
        Xtr, Xte, _ = impute(F.loc[fold.train.index].to_numpy(np.float32), F.loc[fold.test.index].to_numpy(np.float32))
        for d in cells:
            scores[d], _ = run_baseline("B-L2", Xtr, Xte, fold.train, d, a.loc[fold.train.index].to_numpy(),
                                        a.loc[fold.test.index].to_numpy(), None, CFG["seed"])
    else:
        Xtr, Xte, pat = design(fold, names, comp, ref)
        for d in cells:
            scores[d] = fit_scores(Xtr, Xte, fold.train, pat, d, CFG["seed"])
    rows = DEV.long_rows(fold, cells, scores, a, method)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, sep="\t", index=False)
    return f"{fold.key} {method} cells={cells} {time.time() - t0:.1f}s"


def vs_baseline(m: pd.DataFrame, base: str) -> pd.DataFrame:
    """每个方法 X 相对 B-L2 的 cell 级配对差（X − B-L2）：均值、bootstrap 95% CI、胜率、Wilcoxon p。"""
    out = []
    for meth in sorted(set(m.method) - {base}):
        a = SUM.ablation_table(m[m.method.isin([meth, base])], ref=meth)
        out.append(a.assign(method=meth, baseline=base))
    return pd.concat(out) if out else pd.DataFrame(columns=["method", "family"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=CFG["deconv"]["primary"])
    ap.add_argument("--families", default=",".join(EV["families"]))
    ap.add_argument("--compartments", default="GLOM,TUB")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--n-jobs", type=int, default=M2["n_jobs"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    out = ROOT / "results" / "16_deconv" / "eval" / args.reference
    meths = list(EV["methods"]) if args.methods == "all" else args.methods.split(",")
    jobs = []
    for fam in args.families.split(","):
        for comp in args.compartments.split(","):
            folds = make_folds(fam, comp)
            cells = DEV.fold_cells(folds, fam)
            for i, fold in enumerate(folds):
                if not cells[fold.key]:
                    continue
                for mth in meths:
                    p = out / "preds" / mth / f"{fold.key}.tsv"
                    if args.force or not p.exists():
                        jobs.append((mth, fam, comp, i, args.reference, str(p)))
    print(f"{len(jobs)} 个作业", flush=True)
    for msg in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(delayed(run_job)(*j) for j in jobs):
        print(msg, flush=True)
    P = DEV.collect(out)
    P = P[P.family.isin(args.families.split(","))]
    met = metrics_from_preds(P)
    met.to_csv(out / "dev_metrics.tsv", sep="\t", index=False)
    SUM.summary_table(met).to_csv(out / "dev_summary.tsv", sep="\t", index=False)
    vb = vs_baseline(met, EV["baseline"])
    vb.to_csv(out / "vs_BL2.tsv", sep="\t", index=False)
    pairs = [SUM.ablation_table(met[met.method.isin([a, b])], ref=a).assign(method=a, baseline=b)
             for a, b in EV["pairs"] if {a, b} <= set(met.method)]
    if pairs:
        pd.concat(pairs).to_csv(out / "vs_pairs.tsv", sep="\t", index=False)
    if vb.empty:
        return
    show = vb[vb.family.isin(["ALL", "E1", "E2", "E4", "ERCB"])][
        ["method", "family", "n_cells", "auroc_ref", "auroc_alt", "d_auroc_mean", "d_auroc_lo", "d_auroc_hi",
         "d_auroc_frac_ref_better", "adj_auroc_ref", "d_adj_auroc_mean", "d_adj_auroc_lo", "d_adj_auroc_hi",
         "d_adj_auroc_frac_ref_better", "d_adj_auroc_wilcoxon_p"]]
    pd.set_option("display.width", 250)
    print(show.round(3).to_string(index=False))
    write_provenance(f"16_deconv/eval/{args.reference}",
                     [ROOT / "results/16_deconv/run" / EV["run_tag"] / "theta_type.tsv",
                      ROOT / "results/16_deconv/run" / EV["run_tag"] / "theta_state.tsv",
                      ROOT / "data/processed/samples.tsv"],
                     [out / f for f in ("dev_metrics.tsv", "dev_summary.tsv", "vs_BL2.tsv")], CFG["seed"],
                     {"methods": {k: EV["methods"][k] for k in meths}, "families": args.families,
                      "reference": args.reference, "n_cells": int(len(met))})


if __name__ == "__main__":
    main()
