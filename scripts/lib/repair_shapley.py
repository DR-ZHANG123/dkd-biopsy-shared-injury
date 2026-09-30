"""Stage 20：KPMP 组织 pseudobulk 的三层（谱系组成 p × 谱系内状态构成 q × 状态内表达谱 M）Shapley 分解。

组织谱（供体 d）：E_d = Σ_f p_{d,L(f)} · q_{d,f} · M_{d,f}，f = 细粒度状态（SubclassLevel2），L(f) = 谱系。
  谱系 = kpmp.map_to_stage13 的粗类型，但 PT ∪ PT_injured 合为 PT，LOH 拆为 TAL（repair_state.lineages.TAL）与 LOH_rest。
  M = 深度归一的每细胞平均计数（同 stage 19 lib.scp_kpmp.Tissue）；某供体某状态细胞 < min_cells 时用 REF 均值填补。
因子（每个取供体自身值或 REF 供体平均值）：
  p         谱系组成（stage 19「组成」成分）
  q_PT / q_TAL / q_rest       谱系内状态构成（aPT、frPT 等占 PT 的比例 → 「修复失败状态占比上升」）
  M_<谱系>_<组>              状态内表达谱：PT / TAL 各分 normal、repair（a ∪ fr）、other_alt（d、cyc）；其余谱系合为 M_rest
对全部 2^n 个因子子集计算供体分数（SCP 带符号平均秩）与核心基因 log2CPM；供体水平 Shapley 值 φ_{i,d}（Σ_i φ_{i,d} = v_d(全部) − v(无)）。
组间差（A − B）的 Shapley 份额 = 组间 φ 均值差（线性，可做供体标签置换与供体 bootstrap）。
"""
from __future__ import annotations

import itertools
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats

from lib.repair_state import RS, STATES

GROUPS = ("normal", "repair", "other_alt")


def lineage_of(fine: str, coarse: str) -> str:
    if fine in STATES:
        return STATES[fine][0]
    if coarse in ("PT", "PT_injured"):
        return "PT"
    return "LOH_rest" if coarse == "LOH" else coarse


def m_group(fine: str) -> str:
    lin, st = STATES.get(fine, (None, None))
    if lin is None:
        return "M_rest"
    g = "normal" if st == "normal" else "repair" if st in RS["repair_states"][lin] else "other_alt"
    return f"M_{lin}_{g}"


class FineTissue:
    def __init__(self, pb, universe: list[str], ref: pd.Index, min_cells: int, coarse_map: dict):
        c2 = {f: c for c, fs in coarse_map.items() for f in fs}
        rows, X = pb.fine_level(sorted(c2))
        gi = pb.genes.get_indexer(universe)
        self.genes = pd.Index([g for g, i in zip(universe, gi) if i >= 0])
        Xg = X[:, gi[gi >= 0]].toarray().astype(np.float32)
        self.fines = sorted(rows.cell_type.unique())
        self.lin = np.array([lineage_of(f, c2[f]) for f in self.fines])
        self.mgrp = np.array([m_group(f) for f in self.fines])
        self.donors = sorted(rows.donor.unique())
        nf, nd, ng = len(self.fines), len(self.donors), len(self.genes)
        fi = {f: i for i, f in enumerate(self.fines)}
        di = {d: i for i, d in enumerate(self.donors)}
        N = np.zeros((nd, nf), np.float32)
        C = np.zeros((nd, nf, ng), np.float32)
        for r, (d, f, n) in enumerate(zip(rows.donor, rows.cell_type, rows.n_cells)):
            N[di[d], fi[f]] += n
            C[di[d], fi[f]] += Xg[r]
        depth = C.sum((1, 2)) / np.maximum(N.sum(1), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            M = C / N[:, :, None] / depth[:, None, None]
        ok = N >= min_cells
        r = np.isin(self.donors, ref)
        mbar = np.zeros((nf, ng), np.float32)
        self.ref_source = []
        for k in range(nf):                    # 参照谱：REF 合格供体均值 → 全部合格供体均值 → 全部供体合并计数
            if ok[r, k].any():
                mbar[k], src = M[r][ok[r, k], k].mean(0), "REF"
            elif ok[:, k].any():
                mbar[k], src = M[ok[:, k], k].mean(0), "all_donors"
            else:
                mbar[k], src = C[:, k].sum(0) / max(N[:, k].sum(), 1) / depth.mean(), "pooled"
            self.ref_source.append(src)
        del C
        self.mbar = np.nan_to_num(mbar).astype(np.float32)
        self.Mfill = np.where(ok[:, :, None], np.nan_to_num(M), self.mbar[None]).astype(np.float32)
        del M
        self.N, self.ok = N, ok
        self.lins = sorted(set(self.lin))
        Lmask = np.stack([self.lin == L for L in self.lins], 1).astype(np.float32)        # f × L
        NL = N @ Lmask                                                                       # d × L
        self.p = NL / np.maximum(NL.sum(1, keepdims=True), 1)
        self.q = N / np.maximum((NL @ Lmask.T), 1)                                           # d × f
        self.pbar, self.qbar = self.p[r].mean(0), self.q[r].mean(0)
        self.Lmask = Lmask

    def factors(self) -> list[str]:
        qs = [f"q_{L}" for L in ("PT", "TAL")] + ["q_rest"]
        ms = [f"M_{L}_{g}" for L in ("PT", "TAL") for g in GROUPS] + ["M_rest"]
        return ["p"] + qs + [m for m in ms if (self.mgrp == m).any()]

    def _q_factor(self, f_lin: str) -> str:
        return f"q_{f_lin}" if f_lin in ("PT", "TAL") else "q_rest"

    def profile(self, S: set[str]) -> np.ndarray:
        """因子子集 S（取供体值）下的组织谱 d × g（线性）。"""
        nd = len(self.donors)
        p = self.p if "p" in S else np.repeat(self.pbar[None], nd, 0)
        qd = np.array([self._q_factor(L) in S for L in self.lin])
        q = np.where(qd[None], self.q, self.qbar[None])
        W = (p @ self.Lmask.T) * q                                                            # d × f
        md = np.array([g in S for g in self.mgrp])
        Wd = (W * md[None]).astype(np.float32)                                              # 不做花式索引拷贝
        E = np.einsum("df,dfg->dg", Wd, self.Mfill, optimize=False) if md.any() else 0.0
        return E + (W * ~md[None]).astype(np.float32) @ self.mbar


def rank_signed(E: np.ndarray, gi_up: np.ndarray, gi_dn: np.ndarray) -> np.ndarray:
    R = stats.rankdata(E, axis=1) / E.shape[1]
    return R[:, gi_up].mean(1) - R[:, gi_dn].mean(1)


def shapley(T: FineTissue, up: list[str], dn: list[str], genes_out: list[str]) -> tuple[pd.DataFrame, dict]:
    """返回 (供体 × 因子 的分数 Shapley 值, {因子: 供体 × genes_out 的 log2CPM Shapley 值})，另含 'full' / 'state_only' 值。"""
    F = T.factors()
    n = len(F)
    gu, gd = T.genes.get_indexer(up), T.genes.get_indexer(dn)
    gu, gd = gu[gu >= 0], gd[gd >= 0]
    go = T.genes.get_indexer(genes_out)
    go_ok = go >= 0
    V, G = {}, {}
    for k in range(n + 1):
        for S in itertools.combinations(range(n), k):
            E = T.profile({F[i] for i in S})
            V[S] = rank_signed(E, gu, gd)
            L = np.log2(E[:, go[go_ok]] / np.maximum(E.sum(1, keepdims=True), 1e-12) * 1e6 + 1)
            G[S] = L
    w = {k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)}
    phi = np.zeros((len(T.donors), n))
    phig = {F[i]: np.zeros_like(G[()]) for i in range(n)}
    for S in V:
        for i in range(n):
            if i in S:
                continue
            Si = tuple(sorted(S + (i,)))
            phi[:, i] += w[len(S)] * (V[Si] - V[S])
            phig[F[i]] += w[len(S)] * (G[Si] - G[S])
    out = pd.DataFrame(phi, index=T.donors, columns=F)
    full, none = V[tuple(range(n))], V[()]
    out["full_minus_ref"] = full - none
    out["state_only_minus_ref"] = V[tuple(i for i in range(n) if F[i] != "p")] - none
    out["comp_only_minus_ref"] = V[(F.index("p"),)] - none
    out["full_score"] = full
    gnames = [g for g, o in zip(genes_out, go_ok) if o]
    phig = {k: pd.DataFrame(v, index=T.donors, columns=gnames) for k, v in phig.items()}
    phig["full"] = pd.DataFrame(G[tuple(range(n))] - G[()], index=T.donors, columns=gnames)
    return out, phig
