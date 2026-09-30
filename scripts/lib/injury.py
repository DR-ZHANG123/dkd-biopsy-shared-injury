"""通用损伤轴与损伤调整评估。

- full_ranks(): 每个样本在其平台全部测到的基因上的样本内百分位秩（不限于 5,000 节点）。
- injury_axis(): 以「非 DKD 病种 vs 对照」的逐基因效应量在多个来源队列上取平均，得到损伤方向；
  不使用任何 DKD 标签，调用方负责把被评估队列排除在来源之外。
- injury_score(): 损伤轴 top-k 上调基因平均秩 − top-k 下调基因平均秩。
- adjusted_auroc(): Janes & Pepe 协变量调整 AUROC（以阴性组为参照、对协变量做线性位置调整）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.repro import ROOT, load_config

CFG = load_config()
INTERIM = ROOT / CFG["paths"]["interim"]
NON_DISEASE = {"CONTROL", "DKD", "OTHER", "UNKNOWN", "TMD"}


@lru_cache(maxsize=None)
def _series_ranks(series: str) -> pd.DataFrame:
    e = pd.read_parquet(INTERIM / f"{series}_expr.parquet")
    e = e.loc[e.notna().all(axis=1)]
    r = (e.rank(axis=0) - 1) / (e.shape[0] - 1)
    r.columns = [f"{series}|{c}" for c in r.columns]
    return r.T.astype("float32")          # 样本 × 基因


def full_ranks(uids: pd.Index, samples: pd.DataFrame) -> pd.DataFrame:
    parts = [_series_ranks(ser).reindex(idx) for ser, idx in
             pd.Series(uids, index=uids).groupby(samples.loc[uids, "series"])]
    return pd.concat(parts).reindex(uids)


def effect_sizes(R: pd.DataFrame, diag: pd.Series) -> pd.Series | None:
    """非 DKD 病种 vs 对照 的逐基因标准化均值差（Hedges g 近似）。"""
    ctrl, dis = diag == "CONTROL", ~diag.isin(NON_DISEASE)
    if ctrl.sum() < CFG["injury"]["min_controls"] or dis.sum() < CFG["injury"]["min_disease"]:
        return None
    a, b = R[dis.values], R[ctrl.values]
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / (sd + 1e-6)


def injury_axis(source_units: list[pd.Index], samples: pd.DataFrame) -> pd.Series:
    effs = []
    for idx in source_units:
        R = full_ranks(idx, samples).dropna(axis=1)
        g = effect_sizes(R, samples.loc[idx, "diagnosis"])
        if g is not None:
            effs.append(g)
    if not effs:
        raise ValueError("没有满足条件的损伤轴来源队列")
    E = pd.concat(effs, axis=1)
    E = E[E.notna().sum(axis=1) >= max(1, int(np.ceil(CFG["injury"]["min_source_frac"] * E.shape[1])))]
    return E.mean(axis=1, skipna=True).rename("injury_effect")


def injury_score(R: pd.DataFrame, axis: pd.Series) -> pd.Series:
    k = CFG["injury"]["top_k"]
    ax = axis.reindex(R.columns).dropna()
    up, dn = ax.nlargest(k).index, ax.nsmallest(k).index
    return (R[up].mean(axis=1) - R[dn].mean(axis=1)).rename("injury")


def adjusted_auroc(y: np.ndarray, score: np.ndarray, cov: np.ndarray) -> float:
    """在阴性组内拟合 score ~ cov，所有样本取残差后计算 AUROC（协变量调整 AUROC，位置模型）。"""
    neg = y == 0
    X = np.c_[np.ones(neg.sum()), cov[neg]]
    beta = np.linalg.lstsq(X, score[neg], rcond=None)[0]
    resid = score - (beta[0] + beta[1] * cov)
    return float(roc_auc_score(y, resid))


def stratified_auroc(y: np.ndarray, score: np.ndarray, cov: np.ndarray, n_strata: int) -> float:
    q = pd.qcut(cov, n_strata, labels=False, duplicates="drop")
    num = den = 0.0
    for b in np.unique(q):
        yy = y[q == b]
        w = yy.sum() * (len(yy) - yy.sum())
        if w:
            num += w * roc_auc_score(yy, score[q == b])
            den += w
    return float(num / den) if den else float("nan")
