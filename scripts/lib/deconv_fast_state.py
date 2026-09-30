"""Stage 16b 细胞内状态估计器（给定估计的组成 p̂）。两个估计器，假设不同：

state_A（分配式，样本级，不在 bulk 上拟合任何参数）：
  线性化 e = 分位数映射(秩 → 参照组织 log2CPM 分布)；组成期望 ê_g = Σ_k p̂_k S_gk（S = KPMP 训练供体每细胞表达），
  缩放到与 e 同总量；基因残差 r_g = log2(e_g+1) − log2(ê_g+1)。
  假设：bulk 相对组成期望的偏离按各类型对该基因的期望份额 w_gk = p̂_k S_gk / ê_g 分配（BayesPrism 条件期望的
  一阶点估计）。state_A_k = Σ_{g∈M_k} w_gk r_g / Σ_{g∈M_k} w_gk，M_k = 类型 k 的 KPMP 标志基因。
  这等价于「k 的标志程序在组成校正后的对数倍数变化」，权重让被其他类型稀释的标志贡献变小。

state_R（组成回归残差，fold 级）：
  在 fold 训练样本（含对照）上拟合 x_g = β_0g + Σ_k β_gk·log(p̂_k + ε) + ε_g（岭回归），测试 / 训练样本取残差；
  state_R_k = 类型 k 标志基因上的平均残差秩。假设：组成对秩的影响在 log 比例上近似线性、跨队列共享；
  残差中与组成无关的部分即细胞内（程序级）状态。回归只用训练样本（不看测试样本、不用标签）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.deconv_fast_sim import rank_to_linear

EPS = 0.01


def _alloc(R: np.ndarray, P: np.ndarray, S: np.ndarray, ref_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (r：N × G 组成校正对数残差, W：N × G × K 期望份额)。"""
    E = rank_to_linear(R, ref_log)                                  # N × G
    M = P[:, None, :] * S[None, :, :]                               # N × G × K 期望贡献
    Ehat = M.sum(2)
    Ehat = Ehat / Ehat.sum(1, keepdims=True) * E.sum(1, keepdims=True)
    r = np.log2(E + 1) - np.log2(Ehat + 1)
    W = M / np.maximum(M.sum(2, keepdims=True), 1e-12)
    return r, W


def state_alloc_genes(R: np.ndarray, P: np.ndarray, S: np.ndarray, blocks: dict, genes: pd.Index,
                      types: list[str], ref_log: np.ndarray) -> np.ndarray:
    """基因级分配状态 STATE_AG：每类型 k 的特异基因块 B_k 上 w_gk·r_g（= 分配给 k 的对数偏离），按块拼接。"""
    r, W = _alloc(R, P, S, ref_log)
    cols = []
    for k, t in enumerate(types):
        gi = genes.get_indexer(blocks[t])
        cols.append(W[:, gi, k] * r[:, gi])
    return np.concatenate(cols, 1)


def state_alloc(R: np.ndarray, P: np.ndarray, S: np.ndarray, mk: dict, genes: pd.Index, types: list[str],
                ref_log: np.ndarray) -> np.ndarray:
    r, W = _alloc(R, P, S, ref_log)
    out = np.zeros((len(R), len(types)))
    for k, t in enumerate(types):
        gi = genes.get_indexer(mk[t])
        w = W[:, gi, k]
        out[:, k] = (w * r[:, gi]).sum(1) / np.maximum(w.sum(1), 1e-12)
    return out


def logp(P: np.ndarray) -> np.ndarray:
    L = np.log(P + EPS)
    return L - L.mean(1, keepdims=True)                             # 中心化 log 比例（clr 的平滑版）


def resid_genes(Xtr: np.ndarray, Xte: np.ndarray, Ptr: np.ndarray, Pte: np.ndarray,
                ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """组成回归残差（全部基因；STATE_RG）：回归系数只在训练样本上估计。"""
    Ztr, Zte = logp(Ptr), logp(Pte)
    mu = Ztr.mean(0)
    Ztr_c, Zte_c = Ztr - mu, Zte - mu
    xm = Xtr.mean(0)
    B = np.linalg.solve(Ztr_c.T @ Ztr_c + ridge * np.eye(Ztr.shape[1]), Ztr_c.T @ (Xtr - xm))
    return Xtr - xm - Ztr_c @ B, Xte - xm - Zte_c @ B


def state_resid(Xtr: np.ndarray, Xte: np.ndarray, Ptr: np.ndarray, Pte: np.ndarray, mk_idx: list[np.ndarray],
                ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Xtr/Xte：样本 × G 秩；Ptr/Pte：比例；mk_idx：每类型的基因下标。返回 (train, test) × K 状态分数。"""
    Rtr, Rte = resid_genes(Xtr, Xte, Ptr, Pte, ridge)
    return (np.stack([Rtr[:, g].mean(1) for g in mk_idx], 1),
            np.stack([Rte[:, g].mean(1) for g in mk_idx], 1))
