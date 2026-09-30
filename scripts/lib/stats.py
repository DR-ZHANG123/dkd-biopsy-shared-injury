"""stage 09 的统计工具：配对（患者层面）bootstrap AUROC、DerSimonian–Laird 随机效应、校准。"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss


def auc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """S: (m, n) 每行一个模型的分数；y: (n,)。Mann–Whitney 公式（并列取平均秩）。单类时返回 NaN。"""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[0], np.nan)
    r = rankdata(S, axis=1)
    return (r[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot_index(clusters: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """患者层面 bootstrap：按 cluster（dup_group）有放回抽样，返回每次的样本索引。"""
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uc))]
    if all(len(m) == 1 for m in members):
        return list(rng.integers(0, len(clusters), size=(n_boot, len(clusters))))
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), size=len(uc))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def boot_auc(y: np.ndarray, S: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, m)：所有模型共享同一组重抽样索引（配对）。"""
    rng = np.random.default_rng(seed)
    idx = cluster_boot_index(clusters, n_boot, rng)
    return np.stack([auc_rows(y[i], S[:, i]) for i in idx])


def ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def boot_p_two_sided(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(min(1.0, max(p, 1 / len(d))))


def logit(p, clip):
    p = np.clip(p, clip, 1 - clip)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def auc_cc(a, n1: int, n0: int):
    """连续性校正：A_c = (A·n1·n0 + 0.5)/(n1·n0 + 1)，使 AUROC = 0/1 时 logit 有限。"""
    m = n1 * n0
    return (np.asarray(a) * m + 0.5) / (m + 1)


def hanley_mcneil_var(a: float, n1: int, n0: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def logit_auc_var(boot: np.ndarray, a: float, n1: int, n0: int) -> tuple[float, float]:
    """(logit 点估计, 方差)：方差取 bootstrap 方差与 Hanley–McNeil（delta 法）方差的较大者，
    后者作为下限，避免完美分离（bootstrap 方差为 0）时该 fold 被 DL 丢弃或获得无穷权重。"""
    ac = float(auc_cc(a, n1, n0))
    vb = float(np.nanvar(logit(auc_cc(boot, n1, n0), 1e-12), ddof=1)) if np.isfinite(boot).sum() > 1 else 0.0
    vh = hanley_mcneil_var(ac, n1, n0) / (ac * (1 - ac)) ** 2
    return float(logit(ac, 1e-12)), max(vb, vh)


def delta_var(d_boot: np.ndarray, a: float, b: float, n1: int, n0: int) -> float:
    """ΔAUROC 方差：bootstrap 方差，下限为 Hanley–McNeil 方差（两模型平均 AUROC，假定相关 0.5）。"""
    vb = float(np.nanvar(d_boot, ddof=1)) if np.isfinite(d_boot).sum() > 1 else 0.0
    return max(vb, hanley_mcneil_var(float(auc_cc((a + b) / 2, n1, n0)), n1, n0))


def dersimonian_laird(est: np.ndarray, var: np.ndarray) -> dict:
    """随机效应合并；预测区间用 t_{k−2}（Higgins 2009），k<3 时不给。"""
    ok = ~np.isnan(est) & ~np.isnan(var) & (var > 0)
    est, var = est[ok], var[ok]
    k = len(est)
    if k == 0:
        return {"k": 0}
    if k == 1:
        se = float(np.sqrt(var[0]))
        return {"k": 1, "mu": float(est[0]), "se": se, "lo": est[0] - 1.96 * se, "hi": est[0] + 1.96 * se,
                "tau2": np.nan, "I2": np.nan, "Q": np.nan, "pi_lo": np.nan, "pi_hi": np.nan, "p": np.nan}
    w = 1 / var
    mu_f = (w * est).sum() / w.sum()
    Q = float((w * (est - mu_f) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c)
    ws = 1 / (var + tau2)
    mu = float((ws * est).sum() / ws.sum())
    se = float(np.sqrt(1 / ws.sum()))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    if k >= 3:
        t = stats.t.ppf(0.975, k - 2)
        half = t * np.sqrt(tau2 + se ** 2)
        pi = (mu - half, mu + half)
    else:
        pi = (np.nan, np.nan)
    p = float(2 * stats.norm.sf(abs(mu / se))) if se > 0 else np.nan
    return {"k": k, "mu": mu, "se": se, "lo": mu - 1.96 * se, "hi": mu + 1.96 * se, "tau2": tau2,
            "I2": I2, "Q": Q, "pi_lo": pi[0], "pi_hi": pi[1], "p": p}


def calibration(y: np.ndarray, p: np.ndarray, clip: float = 1e-4) -> tuple[float, float]:
    """校准斜率与截距：y ~ logit(p) 的无惩罚 logistic 回归。"""
    import statsmodels.api as sm
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    x = sm.add_constant(logit(p, clip))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        return float(res.params[1]), float(res.params[0])
    except Exception:
        return np.nan, np.nan


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float | None) -> dict:
    out = {"n": len(y), "n_pos": int(y.sum())}
    two = len(np.unique(y)) == 2
    out["auroc"] = float(auc_rows(y, p[None])[0]) if two else np.nan
    out["auprc"] = float(average_precision_score(y, p)) if two else np.nan
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    out["cal_slope"], out["cal_intercept"] = calibration(y, p)
    if thr is not None and not np.isnan(thr):
        pred = p >= thr
        out["threshold"] = thr
        out["sensitivity"] = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
        out["specificity"] = float((~pred[y == 0]).mean()) if (y == 0).any() else np.nan
    return out
