"""Stage 06：线性基线 B1（rank-LASSO）与 B2（rank-PCA(64) + L2 logistic）。

- B1：同一 5,000 节点基因的样本内秩；NaN 用（inner）训练集逐基因中位数填补；L1 logistic，
  inner leave-one-training-cohort-out 选 C。
- B2：PCA 在该 fold 的预训练语料（与 stage 07 完全相同的样本集合，不含测试队列及其重复）上拟合，
  下游 L2 logistic，inner CV 选 C。
- 阈值：inner OOF 预测上的 Youden 指数。label-efficiency 按 config eval.label_fracs 在训练 cohort 内分层抽样
  （抽样种子与 stage 08 相同，保证配对）。
- --inner-only：只做 inner CV（设计迭代用），不对测试队列打分，输出到 results/06_baselines_inner/。

产出：results/06_baselines/predictions.tsv、thresholds.tsv、inner_cv.tsv、PROVENANCE.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # 共享服务器：限制 BLAS 线程，避免 joblib/torch 超额订阅

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import evalkit as E  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()


def run_fold(fold: D.Fold, inner_only: bool) -> tuple[list, list, list]:
    set_global_seed(CFG["seed"])
    samples = D.load_samples_unlabelled()
    folds = D.load_folds()
    tr_all, te = D.downstream_split(fold, CFG, folds, samples)
    corpus = D.pretrain_uids(fold, CFG, samples, folds)
    uids = list(dict.fromkeys(list(tr_all.sample_uid) + list(te.sample_uid) + corpus))
    R = D.load_ranks(uids)
    pca, pmed = E.fit_pca(R.loc[corpus].to_numpy(), CFG["baselines"]["pca_components"], CFG["seed"])
    Xte = R.loc[te.sample_uid].to_numpy()
    Zte = E.pca_transform(pca, pmed, Xte)
    preds, thrs, inner = [], [], []
    base = {"task": fold.task, "compartment": fold.compartment, "fold": fold.fold}
    n_seeds = CFG["probe"]["seeds"]
    for frac in CFG["eval"]["label_fracs"]:
        for seed in E.seeds_for(frac, True, n_seeds):
            tr = D.subsample_labels(tr_all, frac, seed, CFG["eval_ext"]["min_per_class"])
            y, coh = tr.y.to_numpy(), tr.cohort.to_numpy()
            if len(np.unique(y)) < 2:
                continue
            Xtr = R.loc[tr.sample_uid].to_numpy()
            fits = {
                "B1-rankLASSO": (E.fit_logistic_cv(Xtr, y, coh, CFG["baselines"]["lasso_C_grid"], "l1",
                                                   CFG, CFG["seed"] + seed, impute=True), Xte),
                "B2-PCA": (E.fit_logistic_cv(E.pca_transform(pca, pmed, Xtr), y, coh, CFG["probe"]["C_grid"],
                                             "l2", CFG, CFG["seed"] + seed), Zte),
            }
            for mname, (fit, Xt) in fits.items():
                name = E.model_name(mname, frac)
                extra = f"n_nonzero={E.n_selected(fit)}" if mname.startswith("B1") else \
                    f"pca_var={pca.explained_variance_ratio_.sum():.3f}"
                thrs.append({**base, "model": name, "seed": seed, "C": fit["C"], "threshold": fit["threshold"],
                             "inner_auc": fit["inner_auc"], "scheme": fit["scheme"], "n_train": len(tr),
                             "extra": extra})
                for c in np.unique(coh):
                    mk = (coh == c) & ~np.isnan(fit["oof"])
                    if len(np.unique(y[mk])) == 2:
                        inner.append({**base, "model": name, "seed": seed, "inner_cohort": c,
                                      "n": int(mk.sum()), "auroc": roc_auc_score(y[mk], fit["oof"][mk])})
                if not inner_only:
                    p = E.predict_logistic(fit, Xt)
                    preds += [{**base, "model": name, "seed": seed, "sample_uid": u, "y_true": int(t),
                               "score": float(s)} for u, t, s in zip(te.sample_uid, te.y, p)]
    print(f"[06] {fold.key}: train={len(tr_all)} test={len(te)} corpus={len(corpus)}", flush=True)
    return preds, thrs, inner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--compartment")
    ap.add_argument("--fold")
    ap.add_argument("--inner-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    set_global_seed(CFG["seed"])
    stage = "06_baselines_inner" if a.inner_only else "06_baselines"
    out = ROOT / "results" / stage
    out.mkdir(parents=True, exist_ok=True)
    folds = [f for f in D.list_folds(task=a.task, compartment=a.compartment, fold=a.fold)
             if f.task in CFG["tasks"]]
    res = Parallel(n_jobs=min(a.jobs, len(folds)))(delayed(run_fold)(f, a.inner_only) for f in folds)
    preds = pd.DataFrame([r for x in res for r in x[0]], columns=E.PRED_COLS)
    thrs = pd.DataFrame([r for x in res for r in x[1]], columns=E.THR_COLS)
    inner = pd.DataFrame([r for x in res for r in x[2]])
    outputs = []
    if not a.inner_only:
        preds.to_csv(out / "predictions.tsv", sep="\t", index=False)
        outputs.append(out / "predictions.tsv")
    thrs.to_csv(out / "thresholds.tsv", sep="\t", index=False)
    inner.to_csv(out / "inner_cv.tsv", sep="\t", index=False)
    outputs += [out / "thresholds.tsv", out / "inner_cv.tsv"]
    full_run = not (a.task or a.compartment or a.fold)
    if full_run or a.inner_only:
        proc = ROOT / "data" / "processed"
        write_provenance(stage, [proc / "ranks.parquet", proc / "samples.tsv", proc / "folds.tsv",
                                 proc / "genes.txt"], outputs, CFG["seed"],
                         {"n_folds": len(folds), "inner_only": a.inner_only})
    summ = thrs[thrs.model.isin(["B1-rankLASSO", "B2-PCA"])][["task", "compartment", "fold", "model",
                                                              "C", "inner_auc", "scheme", "extra"]]
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
