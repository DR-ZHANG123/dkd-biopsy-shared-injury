"""Stage 21（稳健性补充分析）公共工具。

- units(): 与 stage 11/19 相同的样本表与分析单元（import 11_injury_axis.load_samples）。
- proc_sensitive(): 全基因「取材敏感」判定（健康 KPMP 供体 手术取材 vs 经皮活检，snRNA 组织 pseudobulk 的 Hedges g，
  |g| ≥ proc_min_abs_g 且方向与给定疾病方向相反 = 模拟对照；与 stage 19c 对核心基因的规则相同，推广到任意方向向量）。
- exclude(): 按 gene_variants（full / no_proc / no_ieg / no_proc_ieg）返回要剔除的基因集合。
- two_group(): 两组（样本 × 基因）的 Hedges g（含 J）、抽样方差、均值差（shift）。
- 零分布（均保留基因间相关，生成器分块返回 (g, shift)）：
    perm_null()            疾病 / 对照 标签置换（在存在真实效应时，置换组仍含不等比例的病人 → 方向统计量偏保守）；
    ctrl_split_null()      只在对照内随机拆组（纯噪声协方差，主零分布）；
    resid_rotation_null()  组内残差子空间随机旋转（ROAST 思路；噪声 + 病人间异质性协方差）。
- welch_bh(): 单元内 Welch t（秩尺度）与 BH-FDR。
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.repro import ROOT, load_config
from lib.scp import bh

CFG = load_config()
RV = CFG["robustness21"]
OUT = ROOT / "results" / "21_robustness"
IEG = set(CFG["shared_program"]["kpmp"]["ieg"])
sys.path.insert(0, str(ROOT / "scripts" / "stages"))


def units():
    return import_module("11_injury_axis").load_samples()


def _proc_g() -> pd.Series:
    p = pd.read_csv(ROOT / "results/19_shared_program/kpmp/procurement.tsv.gz", sep="\t",
                    usecols=["dataset", "contrast", "gene", "g"])
    p = p[(p.dataset == "snRNA") & (p.contrast == "REF_surg_vs_REF_perc")]
    return p.set_index("gene").g


PROC_G = None


def proc_sensitive(direction: pd.Series) -> set[str]:
    """direction：基因 → 疾病方向（正 = 疾病中升高）。返回取材敏感且方向模拟对照的基因。"""
    global PROC_G
    if PROC_G is None:
        PROC_G = _proc_g()
    g = PROC_G.reindex(direction.index)
    m = (g.abs() >= RV["proc_min_abs_g"]) & (np.sign(g) == -np.sign(direction))
    return set(direction.index[m.fillna(False).to_numpy(bool)])


def exclude(variant: str, direction: pd.Series) -> set[str]:
    out: set[str] = set()
    if "proc" in variant:
        out |= proc_sensitive(direction)
    if "ieg" in variant:
        out |= IEG
    return out


def core_proc_flags() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/19_shared_program/kpmp/core_procurement.tsv", sep="\t")


def core_exclude(variant: str, comp: str) -> set[str]:
    """核心基因层面：直接沿用 stage 19c 的 procurement_sensitive 标记（同一规则）。"""
    c = core_proc_flags()
    c = c[c.compartment == comp]
    out: set[str] = set()
    if "proc" in variant:
        out |= set(c.gene[c.procurement_sensitive.astype(str) == "True"])
    if "ieg" in variant:
        out |= IEG
    return out


def two_group(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    na, nb = len(A), len(B)
    sd = np.sqrt(((na - 1) * A.var(0, ddof=1) + (nb - 1) * B.var(0, ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    sh = A.mean(0) - B.mean(0)
    g = j * sh / (sd + 1e-6)
    v = (na + nb) / (na * nb) + g ** 2 / (2 * (na + nb))
    return g, v, sh


def _g_from_moments(S1, S2, n1, T1, T2, n):
    """由组 A 的一阶 / 二阶和（S1, S2）与总和（T1, T2）得到 Hedges g 与 shift（逐行一个零分布样本）。"""
    n2 = n - n1
    ma, mb = S1 / n1, (T1 - S1) / n2
    va = (S2 - n1 * ma ** 2) / (n1 - 1)
    vb = ((T2 - S2) - n2 * mb ** 2) / (n2 - 1)
    sd = np.sqrt(np.maximum(((n1 - 1) * va + (n2 - 1) * vb) / (n - 2), 0))
    j = 1 - 3 / (4 * n - 9)
    sh = ma - mb
    return j * sh / (sd + 1e-6), sh


def perm_null(Y: np.ndarray, n1: int, n_perm: int, rng, chunk: int = 100):
    """Y：前 n1 行为疾病组。逐块产出 (g, shift)，每块 ≤ chunk 个置换。"""
    n = len(Y)
    T1, T2 = Y.sum(0), (Y ** 2).sum(0)
    Y2 = Y ** 2
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        A = np.zeros((k, n))
        for i in range(k):
            A[i, rng.choice(n, n1, replace=False)] = 1.0
        yield _g_from_moments(A @ Y, A @ Y2, n1, T1, T2, n)
        done += k


def ctrl_split_null(C: np.ndarray, n1: int, n_perm: int, rng, chunk: int = 100):
    """纯对照零分布：把对照随机分成两组（大小 min(n1, n_c//2) 与其余），产出 (g, shift)。
    不含任何疾病信号，保留对照样本的基因间协方差（技术 / 取材 / 个体差异）；方向统计量（余弦、R²）与尺度无关，
    因此组大小不同不影响其分布形状以外的性质。"""
    nc = len(C)
    k1 = max(2, min(n1, nc // 2))
    T1, T2, C2 = C.sum(0), (C ** 2).sum(0), C ** 2
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        A = np.zeros((k, nc))
        for i in range(k):
            A[i, rng.choice(nc, k1, replace=False)] = 1.0
        yield _g_from_moments(A @ C, A @ C2, k1, T1, T2, nc)
        done += k


def resid_rotation_null(Y: np.ndarray, n1: int, n_rot: int, rng, chunk: int = 100):
    """组内残差旋转零分布：零假设下的效应坐标只由残差子空间（n−2 维，组内中心化后）随机方向生成，
    保留组内（噪声 + 病人间异质性）的基因间协方差；g = 残差方向上的均值差 / 观测合并 SD（尺度无关）。"""
    n = len(Y)
    E = np.vstack([Y[:n1] - Y[:n1].mean(0), Y[n1:] - Y[n1:].mean(0)])
    X = np.c_[np.ones(n), np.r_[np.ones(n1), np.zeros(n - n1)]]
    Q, _ = np.linalg.qr(np.c_[X, rng.standard_normal((n, n - 2))])
    Z = Q[:, 2:].T @ E                                # (n−2) × G：残差坐标
    sd = np.sqrt((E ** 2).sum(0) / (n - 2))
    done = 0
    while done < n_rot:
        k = min(chunk, n_rot - done)
        r = rng.standard_normal((k, n - 2))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        b = r @ Z
        yield b / (sd + 1e-6), b
        done += k


def welch_bh(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, p = stats.ttest_ind(A, B, axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    return t, bh(p)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def null_summary(prefix: str, obs: float, null: np.ndarray) -> dict:
    null = np.asarray(null, float)
    return {f"{prefix}": obs, f"{prefix}_null_median": float(np.median(null)),
            f"{prefix}_null_q95": float(np.quantile(null, 0.95)),
            f"{prefix}_p": float((1 + (null >= obs).sum()) / (1 + len(null)))}
