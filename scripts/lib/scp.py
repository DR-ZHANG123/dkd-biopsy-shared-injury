"""Stage 19：泛 CKD 共享疾病–对照程序（SCP）的公共工具。

- unit_effect(): 单元内「非 DKD 病人 vs 对照」逐基因 Hedges g（含小样本校正 J）与其抽样方差；
  秩 = lib.injury.full_ranks（样本内百分位秩，与 stage 11/13 同口径）。
- re_meta(): 逐基因 DerSimonian–Laird 随机效应合并（向量化），另给 Hartung–Knapp 标准误与 t 检验 P。
- signed_score(): 上调核心基因平均秩 − 下调核心基因平均秩（与 lib.injury.injury_score 同形式）。
- random_set_percentile(): 同大小、同上/下调数的随机基因集零分布上，观测 AUROC 的百分位。
不读任何独立队列；调用方负责来源单元的选择。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from lib.injury import NON_DISEASE, full_ranks
from lib.stats import auc_rows


def hedges(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a、b：样本 × 基因。返回 (g, var(g))；g 已乘 J = 1 − 3/(4(n1+n2)−9)。"""
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    g = j * (a.mean(0) - b.mean(0)) / (sd + 1e-6)
    v = (na + nb) / (na * nb) + g ** 2 / (2 * (na + nb))
    return g, v


def patient_mask(diag: np.ndarray, include_dkd: bool) -> np.ndarray:
    excl = NON_DISEASE if not include_dkd else NON_DISEASE - {"DKD"}
    return ~np.isin(diag, list(excl))


def unit_effect(idx: pd.Index, s: pd.DataFrame, positive: str = "nonDKD") -> pd.DataFrame | None:
    """positive：nonDKD（非 DKD 病人）或 DKD。返回 gene × (g, v, n_pos, n_ctrl)。"""
    R = full_ranks(idx, s).dropna(axis=1)
    d = s.loc[idx, "diagnosis"].to_numpy()
    ctrl = d == "CONTROL"
    pos = patient_mask(d, include_dkd=False) if positive == "nonDKD" else d == "DKD"
    if ctrl.sum() < 3 or pos.sum() < 3:
        return None
    g, v = hedges(R.to_numpy()[pos], R.to_numpy()[ctrl])
    return pd.DataFrame({"g": g, "v": v, "n_pos": int(pos.sum()), "n_ctrl": int(ctrl.sum())}, index=R.columns)


def re_meta(G: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    """G、V：基因 × 来源。逐基因 DL 随机效应；缺失来源按 NaN 跳过。"""
    g, v = G.to_numpy(float), V.to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0)
    k = ok.sum(1)
    w = np.where(ok, 1 / np.where(ok, v, 1), 0.0)
    gz = np.where(ok, g, 0.0)
    sw = w.sum(1)
    mu_f = (w * gz).sum(1) / np.maximum(sw, 1e-12)
    Q = (w * (gz - mu_f[:, None]) ** 2).sum(1)
    c = sw - (w ** 2).sum(1) / np.maximum(sw, 1e-12)
    tau2 = np.maximum(0.0, (Q - (k - 1)) / np.maximum(c, 1e-12))
    ws = np.where(ok, 1 / (np.where(ok, v, 1) + tau2[:, None]), 0.0)
    sws = ws.sum(1)
    mu = (ws * gz).sum(1) / np.maximum(sws, 1e-12)
    se = np.sqrt(1 / np.maximum(sws, 1e-12))
    z = mu / se
    p = 2 * stats.norm.sf(np.abs(z))
    hk = np.sqrt((ws * (gz - mu[:, None]) ** 2).sum(1) / (np.maximum(k - 1, 1) * np.maximum(sws, 1e-12)))
    hk = np.maximum(hk, se * 1e-3)
    p_hk = np.where(k > 1, 2 * stats.t.sf(np.abs(mu / hk), np.maximum(k - 1, 1)), np.nan)
    I2 = np.where(Q > 0, np.maximum(0.0, (Q - (k - 1)) / np.where(Q > 0, Q, 1)), 0.0)
    sgn = np.sign(mu)[:, None]
    sign_frac = np.where(ok, np.sign(g) == sgn, False).sum(1) / np.maximum(k, 1)
    return pd.DataFrame({"g_re": mu, "se": se, "z": z, "p": p, "p_hk": p_hk, "tau2": tau2, "I2": I2, "Q": Q,
                         "k": k, "sign_frac": sign_frac, "g_min_abs": np.where(ok, np.abs(g), np.inf).min(1)},
                        index=G.index)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def signed_score(R: pd.DataFrame, up: list[str], dn: list[str]) -> pd.Series:
    u = [x for x in up if x in R.columns]
    d = [x for x in dn if x in R.columns]
    return (R[u].mean(1) - R[d].mean(1)).rename("scp")


def random_set_null(R: pd.DataFrame, n_up: int, n_dn: int, n_sets: int, rng: np.random.Generator,
                    pool: np.ndarray | None = None) -> np.ndarray:
    """返回 (n_sets, 样本) 的随机集分数矩阵（同大小、同上/下调数；基因从 pool 或 R 全部列中不放回抽取）。"""
    X = R.to_numpy(float)
    cols = np.arange(X.shape[1]) if pool is None else R.columns.get_indexer(pool)
    out = np.empty((n_sets, X.shape[0]))
    for i in range(n_sets):
        pick = rng.choice(cols, n_up + n_dn, replace=False)
        out[i] = X[:, pick[:n_up]].mean(1) - X[:, pick[n_up:]].mean(1)
    return out


def auc_vs_null(y: np.ndarray, obs: np.ndarray, null: np.ndarray) -> dict:
    a = float(auc_rows(y, obs[None, :])[0])
    nl = auc_rows(y, null)
    return {"auroc": a, "null_median": float(np.median(nl)), "null_q95": float(np.quantile(nl, 0.95)),
            "pct_vs_random": float((nl < a).mean() + 0.5 * (nl == a).mean())}


def rho_vs_null(x: np.ndarray, obs: np.ndarray, null: np.ndarray) -> dict:
    r, p = stats.spearmanr(obs, x)
    rx = stats.rankdata(x)
    rn = np.apply_along_axis(stats.rankdata, 1, null)
    rn = rn - rn.mean(1, keepdims=True)
    rx = rx - rx.mean()
    nl = (rn @ rx) / (np.sqrt((rn ** 2).sum(1)) * np.sqrt((rx ** 2).sum()) + 1e-12)
    return {"spearman": float(r), "p": float(p), "null_median": float(np.median(nl)),
            "pct_vs_random": float((nl < r).mean())}


def clr(P: pd.DataFrame, eps: float) -> pd.DataFrame:
    L = np.log(P.clip(lower=eps))
    return L.sub(L.mean(1), axis=0)
