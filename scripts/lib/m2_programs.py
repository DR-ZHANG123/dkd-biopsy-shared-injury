"""细胞类型程序先验：KPMP 健康参考标志基因（stage 14；可选 stage 13 旧标志作对照）沿基因图扩散得到的软成员矩阵 M（基因 × 程序）。

M_c = Σ_{k=0..K} β^k Â^k m_c，m_c = 标志基因均匀向量，Â = D^-1/2 (A + I) D^-1/2（无权：
stage 05 两类边权尺度不同——STRING score 与共表达 |r|——无权化使 coexpr / ppi / rewired 三个图变体可比）。
每列只保留 top program_size 个基因并归一化为列和 1，程序分数 = 程序内基因秩的加权平均。
graph = none 时 M = 标志基因均匀权重（去图消融）；rewired = 在共表达诱导子图上保度数重连（lib.data.rewire_induced）。
M 只依赖图与单细胞对照标志，不依赖任何诊断标签。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.data import load_edges, rewire_induced
from lib.repro import ROOT, load_config

CFG = load_config()
MK = CFG["model2"]["markers"]


@lru_cache(maxsize=None)
def load_markers(source: str) -> pd.DataFrame:
    """细胞类型标志（cell_type, gene）。kpmp：stage 14 KPMP 健康参考（snRNA ∪ scRNA；内皮拆为肾小球毛细血管
    ENDO_GC / 小动脉 ENDO_ART / 管周 ENDO_PT，去掉笼统的 ENDO）；stage13：旧的两套对照单细胞标志（对照用）。"""
    if source == "stage13":
        return pd.read_csv(ROOT / "results/13_celltype/celltype_markers.tsv", sep="\t")[["cell_type", "gene"]]
    m = pd.read_csv(ROOT / MK["kpmp_file"], sep="\t")
    m = m[m.reference.isin(MK["kpmp_references"]) & ~m.cell_type.isin(MK["kpmp_drop"])]
    return m[["cell_type", "gene"]].drop_duplicates().reset_index(drop=True)


def _adjacency(graph: str, genes: tuple[str, ...], rewire_seed: int) -> sp.csr_matrix | None:
    n = len(genes)
    if graph == "none":
        return None
    kind = {"coexpr": "coexpr", "ppi": "ppi", "rewired": "coexpr"}[graph]
    ei = load_edges(kind, list(genes))
    if graph == "rewired":
        ei = rewire_induced(ei, n, 1000 * rewire_seed + 7, CFG["graphs"]["rewire_swap_factor"])
    ei = ei.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + sp.eye(n, dtype=np.float32, format="csr")
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    return (Dm @ A @ Dm).tocsr()


@lru_cache(maxsize=None)
def program_matrix(graph: str, genes: tuple[str, ...], hops: int, beta: float, size: int,
                   rewire_seed: int = 0, marker_source: str = "kpmp") -> tuple[np.ndarray, list[str]]:
    """返回 (M: G × C float32, 程序名列表)。"""
    mk = load_markers(marker_source)
    gi = {g: i for i, g in enumerate(genes)}
    names, cols = [], []
    for c, g in mk.groupby("cell_type", sort=True):
        idx = [gi[x] for x in g.gene if x in gi]
        if len(idx) < 3:
            continue
        v = np.zeros(len(genes), np.float32)
        v[idx] = 1.0 / len(idx)
        names.append(c)
        cols.append(v)
    M0 = np.stack(cols, 1)
    A = _adjacency(graph, genes, rewire_seed)
    if A is None:
        return M0, names
    acc, cur = M0.copy(), M0.copy()
    for k in range(1, hops + 1):
        cur = A @ cur
        acc += (beta ** k) * cur
    M = np.zeros_like(acc)
    for j in range(acc.shape[1]):
        top = np.argsort(-acc[:, j])[:size]
        top = top[acc[top, j] > 0]
        M[top, j] = acc[top, j]
    M /= M.sum(0, keepdims=True)
    return M.astype(np.float32), names


def program_table(M: np.ndarray, names: list[str], genes: list[str], seeds: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    mk = load_markers("kpmp") if seeds is None else seeds
    is_seed = set(zip(mk.cell_type, mk.gene))
    rows = []
    for j, c in enumerate(names):
        nz = np.nonzero(M[:, j])[0]
        for i in nz[np.argsort(-M[nz, j])]:
            rows.append({"program": c, "gene": genes[i], "weight": float(M[i, j]),
                         "is_marker": (c, genes[i]) in is_seed})
    return pd.DataFrame(rows)
