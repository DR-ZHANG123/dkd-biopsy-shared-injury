"""下游线性模型的共用部件：inner CV 选 C、inner OOF Youden 阈值、NaN 中位数填补。

inner CV 规则（config inner_cv）：训练 cohort 数 ≥ min_cohorts 且每个留出 cohort 之外的训练部分
两类都有样本时，用 leave-one-training-cohort-out；否则按 (cohort, y) 分层 k 折。
选 C 的依据是 pooled OOF AUROC（并列时取更小的 C，即更强正则）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def median_impute_fit(X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.5, med).astype(np.float32)   # 训练集全缺失的基因取中位秩 0.5


def median_impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), med[None, :], X).astype(np.float32)


def inner_splits(cohorts: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> tuple[str, list]:
    ic = cfg["inner_cv"]
    uc = np.unique(cohorts)
    if len(uc) >= ic["min_cohorts"]:
        splits = []
        for c in uc:
            va = np.where(cohorts == c)[0]
            tr = np.where(cohorts != c)[0]
            if len(np.unique(y[tr])) < 2:
                splits = []
                break
            splits.append((tr, va))
        if splits:
            return "loco", splits
    strat = pd.Series(cohorts).astype(str) + "|" + pd.Series(y).astype(str)
    counts = strat.value_counts()
    k = int(min(ic["fallback_kfold"], max(2, np.bincount(y).min())))
    lab = np.where(strat.map(counts).to_numpy() >= k, strat.to_numpy(), y.astype(str))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return f"strat{k}", list(skf.split(np.zeros(len(y)), lab))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(min(thr[i], 1.0))


def _make_lr(penalty: str, C: float, cfg: dict, seed: int):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    lr = LogisticRegression(penalty=penalty, C=C, solver=solver, class_weight="balanced",
                            max_iter=cfg["baselines"]["max_iter"], random_state=seed)
    steps = [StandardScaler()] if cfg["baselines"]["standardize"] else []
    return make_pipeline(*steps, lr)


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


def _pooled_auc(y, p) -> float:
    ok = ~np.isnan(p)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], p[ok]))


def fit_logistic_cv(X: np.ndarray, y: np.ndarray, cohorts: np.ndarray, C_grid: list[float],
                    penalty: str, cfg: dict, seed: int, impute: bool = False) -> dict:
    """inner CV 选 C → 全训练集重拟合 → inner OOF Youden 阈值。

    impute=True 时 NaN 用每个（内层）训练集的逐列中位数填补 —— 填补值也在 inner 训练折内估计。
    返回 dict：model, med（外层填补值或 None）, C, threshold, inner_auc, scheme, oof, cv_table。
    """
    scheme, splits = inner_splits(cohorts, y, cfg, seed)
    rows, oofs = [], {}
    for C in C_grid:
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            if len(np.unique(y[tr])) < 2:          # 该内层训练折只有一类：此折 OOF 留空
                continue
            Xtr, Xva = X[tr], X[va]
            if impute:
                med = median_impute_fit(Xtr)
                Xtr, Xva = median_impute_apply(Xtr, med), median_impute_apply(Xva, med)
            m = _fit(_make_lr(penalty, C, cfg, seed), Xtr, y[tr])
            oof[va] = m.predict_proba(Xva)[:, 1]
        oofs[C] = oof
        rows.append({"C": C, "inner_auc": _pooled_auc(y, oof)})
    tab = pd.DataFrame(rows)
    best = tab.fillna(-1).sort_values(["inner_auc", "C"], ascending=[False, True]).iloc[0]
    C = float(best.C)
    med = median_impute_fit(X) if impute else None
    Xf = median_impute_apply(X, med) if impute else X
    model = _fit(_make_lr(penalty, C, cfg, seed), Xf, y)
    oof = oofs[C]
    ok = ~np.isnan(oof)
    return {"model": model, "med": med, "C": C, "threshold": youden_threshold(y[ok], oof[ok]),
            "inner_auc": float(best.inner_auc), "scheme": scheme, "oof": oof, "cv_table": tab}


def predict_logistic(fit: dict, X: np.ndarray) -> np.ndarray:
    if fit["med"] is not None:
        X = median_impute_apply(X, fit["med"])
    return fit["model"].predict_proba(X)[:, 1]


def n_selected(fit: dict) -> int:
    coef = fit["model"][-1].coef_
    return int((np.abs(coef) > 1e-10).sum())


def fit_pca(Xcorpus: np.ndarray, n_comp: int, seed: int) -> tuple[PCA, np.ndarray]:
    """PCA 在预训练语料上拟合；NaN 用语料中位数填补（该中位数随 PCA 一起保存，用于下游样本）。"""
    med = median_impute_fit(Xcorpus)
    pca = PCA(n_components=min(n_comp, *Xcorpus.shape), random_state=seed, svd_solver="full")
    pca.fit(median_impute_apply(Xcorpus, med))
    return pca, med


def pca_transform(pca: PCA, med: np.ndarray, X: np.ndarray) -> np.ndarray:
    return pca.transform(median_impute_apply(X, med)).astype(np.float32)


def seeds_for(frac: float, deterministic: bool, n_seeds: int) -> list[int]:
    """确定性模型在全标签下只有一个有效种子；抽样或随机训练时用全部种子。"""
    return [0] if (frac >= 1.0 and deterministic) else list(range(n_seeds))


def model_name(base: str, frac: float) -> str:
    return base if frac >= 1.0 else f"{base}@lf{frac:g}"


PRED_COLS = ["task", "compartment", "fold", "model", "seed", "sample_uid", "y_true", "score"]
THR_COLS = ["task", "compartment", "fold", "model", "seed", "C", "threshold", "inner_auc", "scheme",
            "n_train", "extra"]
