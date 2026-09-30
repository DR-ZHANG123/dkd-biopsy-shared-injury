"""Stage 15 对照：同 fold、同训练样本、同基因宇宙与填补，每个病种 d 单独 one-vs-rest（训练 = 训练集病人）。

- B-rankLASSO / B-L2：全部训练病人合并，秩特征（训练集方差 top baseline_max_genes 基因）。
- B-L2-unitcenter：训练特征按 stratum 中心化（减去该 stratum 训练病人的逐基因均值）；测试不中心化——
  线性分数在测试单元内加常数平移不改变单元内 AUROC。
- B-L2-ucmix：同上，但只用同时含 d 阳性与阴性病人的 stratum（最接近常规「批内训练」做法）。
- B-cPCA：对比 PCA（Abid 2018）：target = 训练病人、background = 训练对照（均按 stratum 中心化），
  C_t − αC_b 的前 k 个特征向量在数据张成子空间内求（QR 后小矩阵特征分解），投影后 L2；α 由 inner AUROC 选。
- B-prog-L2：固定细胞类型程序分数（与 RRG-ID-fixedProg 相同的 M）+ stratum 中心化 L2。
- B2-PCA：项目主对照 B2 的同 fold 版本：PCA（baselines.pca_components）在 fold 训练样本上无监督拟合 + L2 logistic。
- B-injury：直接用共享轴分数 a，按训练集 d vs 其他病人的均值差定向。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evalkit import fit_logistic_cv


def predict_logistic(fit: dict, X: np.ndarray) -> np.ndarray:
    """返回 logit（decision_function）而非概率：训练按 stratum 中心化、测试不中心化时，
    概率会饱和成并列（AUROC 退化为 0.5），logit 保持单元内排序。"""
    return fit["model"].decision_function(X)
from lib.repro import load_config

CFG = load_config()
M2 = CFG["model2"]


def center_groups(X: np.ndarray, grp: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """按组中心化；ref 为参与求均值的样本布尔（默认全部）。"""
    out = X.copy()
    ref = np.ones(len(X), bool) if ref is None else ref
    for g in np.unique(grp):
        m = grp == g
        r = m & ref
        out[m] -= (X[r] if r.any() else X[m]).mean(0)
    return out


def _top_var(X: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-X.var(0))[:min(k, X.shape[1])]


def _lr(X, y, units, penalty, grid, seed):
    return fit_logistic_cv(X, y, units, grid, penalty, CFG, seed)


def _cpca_basis(T: np.ndarray, Bk: np.ndarray, alpha: float, k: int) -> np.ndarray:
    B = np.vstack([T / np.sqrt(len(T)), Bk * np.sqrt(alpha / max(len(Bk), 1))])
    J = np.r_[np.ones(len(T)), -np.ones(len(Bk))]
    Q, _ = np.linalg.qr(B.T)                      # G × m
    R = B @ Q                                     # m × m：C = Qᵀ Bᵀ J B Q
    Cs = R.T @ (J[:, None] * R)
    w, v = np.linalg.eigh(Cs)
    return Q @ v[:, np.argsort(-w)[:k]]           # G × k


def run_baseline(name: str, Xtr: np.ndarray, Xte: np.ndarray, train: pd.DataFrame, d: str,
                 a_tr: np.ndarray, a_te: np.ndarray, M: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    pat = train.role.isin(["head", "neg"]).to_numpy()
    y_all = (train.diagnosis.to_numpy() == d).astype(int)
    grp = (train.stratum if M2["baseline_center_level"] == "stratum" else train.unit).to_numpy()
    units = train.unit.to_numpy()
    bl = CFG["baselines"]
    if name == "B-injury":
        sgn = np.sign(a_tr[pat & (y_all == 1)].mean() - a_tr[pat & (y_all == 0)].mean()) or 1.0
        return sgn * a_te, {"sign": float(sgn)}
    if name == "B-prog-L2":
        Ftr, Fte = Xtr @ M, Xte @ M
        Ftr = center_groups(Ftr, grp, pat)
        f = _lr(Ftr[pat], y_all[pat], units[pat], "l2", CFG["probe"]["C_grid"], seed)
        return predict_logistic(f, Fte), {"C": f["C"], "inner_auc": f["inner_auc"]}
    g = _top_var(Xtr[pat], M2["baseline_max_genes"])
    Xtr, Xte = Xtr[:, g], Xte[:, g]
    if name == "B-rankLASSO":
        f = _lr(Xtr[pat], y_all[pat], units[pat], "l1", bl["lasso_C_grid"], seed)
    elif name == "B-L2":
        f = _lr(Xtr[pat], y_all[pat], units[pat], "l2", CFG["probe"]["C_grid"], seed)
    elif name in ("B-L2-unitcenter", "B-L2-ucmix"):
        Xc = center_groups(Xtr, grp, pat)
        use = pat.copy()
        if name == "B-L2-ucmix":
            ok = [s for s in np.unique(grp) if (y_all[pat & (grp == s)] == 1).any()
                  and (y_all[pat & (grp == s)] == 0).any()]
            use &= np.isin(grp, ok)
        if y_all[use].sum() < 2 or (1 - y_all[use]).sum() < 2:
            return np.full(len(Xte), np.nan), {"skipped": "too_few"}
        f = _lr(Xc[use], y_all[use], units[use], "l2", CFG["probe"]["C_grid"], seed)
    elif name == "B2-PCA":
        from lib.evalkit import fit_pca, pca_transform
        pca, med = fit_pca(Xtr, bl["pca_components"], seed)   # 无监督：fold 训练样本（含对照）上拟合
        Ztr, Zte = pca_transform(pca, med, Xtr), pca_transform(pca, med, Xte)
        f = _lr(Ztr[pat], y_all[pat], units[pat], "l2", CFG["probe"]["C_grid"], seed)
        return predict_logistic(f, Zte), {"C": f["C"], "inner_auc": f["inner_auc"]}
    elif name == "B-cPCA":
        ctrl = train.role.to_numpy() == "ctrl"
        Xc = center_groups(Xtr, grp, pat)
        Bc = center_groups(Xtr, grp, ctrl)
        best = None
        for al in M2["cpca_alphas"]:
            V = _cpca_basis(Xc[pat], Bc[ctrl], al, M2["cpca_k"])
            f = _lr(Xc[pat] @ V, y_all[pat], units[pat], "l2", CFG["probe"]["C_grid"], seed)
            if best is None or f["inner_auc"] > best[1]["inner_auc"]:
                best = (V, f, al)
        V, f, al = best
        return predict_logistic(f, Xte @ V), {"C": f["C"], "alpha": al, "inner_auc": f["inner_auc"]}
    else:
        raise ValueError(name)
    return predict_logistic(f, Xte), {"C": f["C"], "inner_auc": f["inner_auc"]}
