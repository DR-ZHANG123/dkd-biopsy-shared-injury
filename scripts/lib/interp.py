"""Stage 18：冻结 RRG-ID 的线性等效权重与稳定性度量。

冻结设计（z 线性、固定程序、gene_resid、无自由程序）下，第 s 个种子的 head d 分数对输入是精确线性的：
    h_{d,s}(x) = β_{d,s} · (x M − μ_s) / σ_s + u_{d,s} · (x − 0.5),   β_{d,s} = (V_s A_s)_d
集成分数（stage 15 fit_predict）= 种子均值 of (h_{d,s} − m_{d,s}) / sd_{d,s}，m、sd 为训练集 head 分数的均值 / SD。
因此定义（均为「集成分数单位」，即训练集 head 分数 SD）：
  程序权重    b_{d,c}   = mean_s β_{d,s,c} / sd_{d,s}                   —— 程序分数每变化 1 个训练 SD 的效应（偏效应）
  基因程序路径 gp_{d,g} = mean_s Σ_c M_{g,c} β_{d,s,c} / (σ_{s,c} sd_{d,s})
  基因残差路径 gr_{d,g} = mean_s u_{d,s,g} / sd_{d,s}
  基因总效应  w_{d,g}   = gp + gr                                        —— ∂(集成分数)/∂x_g，x 为样本内秩（0–1）
  标准化总效应 e_{d,g}  = w_{d,g} · SD_train(x_g)                         —— 梯度 × 输入尺度：基因变化 1 个 SD 的效应
线性模型中「梯度 × 输入」的样本平均绝对贡献 = |w| · mean|x − x̄|，与 e 同序（差一个近似常数），这里用 SD 版本。
另给结构系数（边际关联）：训练病人中，stratum 内中心化后的 程序分数 / 基因秩 与 集成 head 分数的 Pearson r。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def model_weights(m: dict, X: np.ndarray) -> dict:
    """m：fit_predict(keep_models=True) 存下的单个种子参数；X：训练集（已插补）秩矩阵。"""
    A, V, W, U = (m[k].double().numpy() for k in ("A", "V", "W_prog", "U"))
    if U.size == 0:                                         # gene_resid = false（诊断变体）
        U = np.zeros((V.shape[0], W.shape[0]))
    mu, sd = m["p_mu"].double().numpy(), m["p_sd"].double().numpy()
    beta = V @ A                                            # heads × C（标准化程序单位）
    P = (X.astype(np.float64) @ W - mu) / sd
    h = P @ beta.T + (X.astype(np.float64) - 0.5) @ U.T     # 与 RRGID.forward(eval) 相同
    h_mu, h_sd = h.mean(0), h.std(0, ddof=1) + 1e-6         # 与 fit_predict 的 ht.mean(0) / ht.std(0) 相同
    gp = W @ (beta / sd).T / h_sd                           # G × heads
    gr = U.T / h_sd
    return {"beta": (beta / h_sd[:, None]).astype(np.float32), "gp": gp.astype(np.float32),
            "gr": gr.astype(np.float32), "h_mu": h_mu, "h_sd": h_sd,
            "score_train": ((h - h_mu) / h_sd).astype(np.float32)}


def centered(Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        Y[m] -= Y[m].mean(0)
    return Y


def structure_corr(Xc: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Xc：n × p（已中心化），sc：n × heads（已中心化）→ p × heads 的 Pearson r。"""
    xs = np.sqrt((Xc ** 2).sum(0)) + 1e-12
    ss = np.sqrt((sc ** 2).sum(0)) + 1e-12
    return (Xc.T @ sc) / xs[:, None] / ss[None, :]


# ---------------------------------------------------------------- 稳定性
def pairwise_spearman(M: pd.DataFrame) -> np.ndarray:
    """M：特征 × 模型；返回所有模型对的 Spearman（上三角）。"""
    M = M.dropna(axis=1, how="all")
    r = spearmanr(M.values, nan_policy="omit")[0]
    r = np.atleast_2d(r)
    if M.shape[1] == 2:
        return np.array([float(r)]) if np.ndim(r) == 0 else np.array([r[0, 1]])
    iu = np.triu_indices(M.shape[1], 1)
    return r[iu]


def topk_sets(M: pd.DataFrame, k: int) -> dict[str, set]:
    return {c: set(M[c].abs().nlargest(k).index) for c in M.columns if M[c].notna().any()}


def pairwise_jaccard(sets: dict[str, set]) -> np.ndarray:
    keys = list(sets)
    out = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
    return np.array(out)


def sign_consistency(M: pd.DataFrame, ref: pd.Series) -> pd.Series:
    """每个特征：与参考符号（共识）相同的模型比例（忽略 NaN）。"""
    s = np.sign(M).mul(np.sign(ref), axis=0)
    return (s > 0).sum(1) / M.notna().sum(1).clip(lower=1)


def topk_freq(M: pd.DataFrame, k: int) -> pd.Series:
    sets = topk_sets(M, k)
    cnt = pd.Series(0.0, index=M.index)
    for s in sets.values():
        cnt[list(s)] += 1
    return cnt / max(1, len(sets))


def signed_set_score(R: pd.DataFrame, genes: pd.Series) -> np.ndarray:
    """R：样本 × 基因 秩；genes：基因 → 符号（±1）。分数 = Σ sign·rank / n（同 stage 12 score_sig）。"""
    g = [x for x in genes.index if x in R.columns]
    w = genes[g].to_numpy(float)
    return R[g].to_numpy() @ w / len(g)


# ---------------------------------------------------------------- 程序层面的总效应投影
def program_projection(w: np.ndarray, e: np.ndarray, M: np.ndarray, p_sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w：G 基因总效应（∂分数/∂秩），e：G 标准化总效应，M：G × C 程序矩阵（列 = 标志基因均匀权重）。

    pi_c = SD(p_c) · Σ_{g∈c} w_g：程序 c 全部标志基因同步移动、使程序分数改变 1 个训练 SD 时集成分数的变化
          （程序通路 + 残差通路一并计入；冻结模型中程序通路贡献 < 1%，见 program_weights.tsv）；
    z_c  = 标志基因上 e 的均值相对全宇宙的解析 z（无放回抽样的有限总体校正）：该细胞类型标志是否被系统性加权。
    """
    S = M > 0
    n = S.sum(0)
    pi = p_sd * (S.T.astype(np.float64) @ w)
    G = len(e)
    mu, sd = e.mean(), e.std(ddof=1)
    me = (S.T.astype(np.float64) @ e) / n
    z = (me - mu) / (sd / np.sqrt(n) * np.sqrt((G - n) / (G - 1)))
    return pi, z
