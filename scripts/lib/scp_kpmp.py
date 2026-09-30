"""Stage 19c 用：KPMP 供体 × 细胞类型 pseudobulk 上的 SCP 检验工具（供体为重复单位）。

- donor_groups(): REF / CKD（DKD ∪ OTHER_CKD）/ 经皮活检子集 / 健康手术取材（取材对照）。
- type_de(): 某细胞类型内 A vs B 的逐基因 log2FC、Hedges g、供体标签置换 P（双侧）。
- Tissue：用 供体 × 类型 的每细胞平均计数重建组织 pseudobulk，并构造反事实组织：
    full        供体自身组成 × 自身类型谱（= 真实组织 pseudobulk）
    comp_only   供体组成 × 健康参考平均类型谱（只变组成）
    state_only  健康参考平均组成 × 供体自身类型谱（只变细胞内状态）
    state_<k>   参考组成 × 仅类型 k 用供体自身谱（其余用参考谱）
    comp_<k>    参考组成但类型 k 的比例换成供体自身（重归一）× 参考谱
  组织分数 = SCP 带符号平均秩（样本内秩，基因宇宙 = 该区室 bulk meta 基因 ∩ KPMP 基因）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from lib.kpmp_stats import hedges_g, label_perms, log2cpm


def donor_groups(pb, spk: dict) -> dict[str, pd.Index]:
    col = spk["procurement_col"][pb.name]
    perc = pb.donors.index[pb.donors[col].astype(str) == spk["percutaneous"]]
    ref = pb.group_donors("REF")
    ckd = pb.group_donors(spk["ckd_groups"][0])
    for g in spk["ckd_groups"][1:]:
        ckd = ckd.union(pb.group_donors(g))
    return {"REF": ref, "CKD": ckd, "DKD": pb.group_donors("DKD"), "OTHER_CKD": pb.group_donors("OTHER_CKD"),
            "REF_perc": ref.intersection(perc), "CKD_perc": ckd.intersection(perc), "REF_surg": ref.difference(perc)}


def perm_lfc(L: np.ndarray, n_a: int, n_perm: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """L：供体 × 基因（前 n_a 行为 A 组）。返回观测 log2FC 与双侧置换 P。"""
    assign = label_perms(L.shape[0], n_a, n_perm, rng)
    A = assign / assign.sum(1, keepdims=True)
    B = ~assign / (~assign).sum(1, keepdims=True)
    D = (A - B) @ L
    obs = D[0]
    p = (1 + (np.abs(D[1:]) >= np.abs(obs) - 1e-12).sum(0)) / (1 + len(D) - 1)
    return obs, p


def type_de(L: np.ndarray, ia: np.ndarray, ib: np.ndarray, n_perm: int, rng) -> dict[str, np.ndarray]:
    X = np.vstack([L[ia], L[ib]])
    lfc, p = perm_lfc(X, len(ia), n_perm, rng)
    return {"log2fc": lfc, "g": hedges_g(L[ia], L[ib]), "p_perm": p}


class Tissue:
    """rows/X：aggregate() 输出（供体 × 类型计数）。"""

    def __init__(self, rows: pd.DataFrame, X: sp.csr_matrix, genes: pd.Index, universe: list[str], ref: pd.Index,
                 min_cells: int):
        gi = genes.get_indexer(universe)
        self.genes = pd.Index([g for g, i in zip(universe, gi) if i >= 0])
        Xg = X[:, gi[gi >= 0]].toarray().astype(float)
        self.types = sorted(rows.cell_type.unique())
        self.donors = sorted(rows.donor.unique())
        nt, nd, ng = len(self.types), len(self.donors), len(self.genes)
        ti = {t: i for i, t in enumerate(self.types)}
        di = {d: i for i, d in enumerate(self.donors)}
        self.N = np.zeros((nd, nt))
        self.C = np.zeros((nd, nt, ng))
        for r, (d, t, n) in enumerate(zip(rows.donor, rows.cell_type, rows.n_cells)):
            self.N[di[d], ti[t]] += n
            self.C[di[d], ti[t]] += Xg[r]
        tot_cells = self.N.sum(1)
        depth = self.C.sum((1, 2)) / np.maximum(tot_cells, 1)             # 每细胞平均计数（全部类型）
        self.P = self.N / np.maximum(tot_cells[:, None], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.M = self.C / self.N[:, :, None] / depth[:, None, None]    # 深度归一的每细胞平均谱
        self.ok = self.N >= min_cells
        r = np.isin(self.donors, ref)
        self.pbar = self.P[r].mean(0)
        self.mbar = np.stack([np.nanmean(np.where(self.ok[r, k, None], self.M[r, k], np.nan), 0)
                              for k in range(nt)])
        self.mbar = np.nan_to_num(self.mbar)
        self.Mfill = np.where(self.ok[:, :, None], np.nan_to_num(self.M), self.mbar[None])

    def profiles(self) -> dict[str, np.ndarray]:
        """场景 → 供体 × 基因 线性表达。"""
        out = {"full": self.C.sum(1) / np.maximum(self.N.sum(1), 1)[:, None],
               "comp_only": self.P @ self.mbar,
               "state_only": np.einsum("k,dkg->dg", self.pbar, self.Mfill)}
        for k, t in enumerate(self.types):
            Mk = np.repeat(self.mbar[None], len(self.donors), 0)
            Mk[:, k] = self.Mfill[:, k]
            out[f"state_{t}"] = np.einsum("k,dkg->dg", self.pbar, Mk)
            Pk = np.repeat(self.pbar[None], len(self.donors), 0)
            Pk[:, k] = self.P[:, k]
            Pk = Pk / Pk.sum(1, keepdims=True)
            out[f"comp_{t}"] = Pk @ self.mbar
        return out

    def score(self, E: np.ndarray, up: list[str], dn: list[str]) -> np.ndarray:
        R = stats.rankdata(E, axis=1) / E.shape[1]
        u, d = self.genes.get_indexer(up), self.genes.get_indexer(dn)
        return R[:, u[u >= 0]].mean(1) - R[:, d[d >= 0]].mean(1)

    def tissue_log2cpm(self) -> pd.DataFrame:
        return pd.DataFrame(log2cpm(self.C.sum(1)), index=self.donors, columns=self.genes)
