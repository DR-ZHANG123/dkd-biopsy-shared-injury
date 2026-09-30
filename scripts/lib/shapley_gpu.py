"""Stage 21：可自定义因子划分的 KPMP 组织 Shapley 分解（GPU 批量；A3 / A6 / 复算校验）。

在 lib.repair_shapley.FineTissue（stage 20c 的组织谱与参照）之上，允许把谱系内状态构成 q 按状态拆成多个因子：
  PT / TAL 谱系：每个 q 因子拥有一组非正常细粒度状态（取供体值，否则取 REF 均值）；正常细胞总份额 = 1 − 非正常份额之和，
  正常细胞内部亚型构成（S1/S2/S3 等）归 owner 因子所有。所有 q 因子都取供体值时恰好还原供体自身 q；都不取时还原 REF 均值。
  其余谱系：q_rest 整体取供体值或 REF 均值（同 stage 20c）。
  M 因子：细粒度状态集合，取供体自身谱（Mfill）或 REF 均值谱（mbar）。
分数 = SCP 带符号平均秩（rankdata(E)/n，并列取平均），与 stage 20c 相同；全部 2^n 个子集在 GPU 上分批计算。
"""
from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd
import torch

from lib.repair_shapley import FineTissue
from lib.repair_state import RS, STATES


def fine_state(T: FineTissue) -> list[str | None]:
    return [STATES[f][1] if f in STATES else None for f in T.fines]


def spec_original(T: FineTissue) -> dict:
    """与 stage 20c 完全相同的 11 因子划分（用于校验）。"""
    st = fine_state(T)
    spec = {"p": ("p",)}
    for L in ("PT", "TAL"):
        spec[f"q_{L}"] = ("q", [i for i, (l, s) in enumerate(zip(T.lin, st)) if l == L and s != "normal"], L)
    spec["q_rest"] = ("qrest", [i for i, l in enumerate(T.lin) if l not in ("PT", "TAL")])
    for g in sorted(set(T.mgrp)):
        spec[g] = ("m", list(np.flatnonzero(T.mgrp == g)))
    return spec


def spec_split(T: FineTissue) -> dict:
    """A6：adaptive（aPT、aTAL）与 failed-repair（frPT、frTAL）的占比与表达谱各自为独立因子。"""
    st = fine_state(T)
    spec = {"p": ("p",)}
    for L in ("PT", "TAL"):
        a, fr = RS["repair_states"][L]
        ii = [i for i, l in enumerate(T.lin) if l == L]
        spec[f"q_{a}"] = ("q", [i for i in ii if st[i] == a], None)
        spec[f"q_{fr}"] = ("q", [i for i in ii if st[i] == fr], None)
        spec[f"q_{L}_other"] = ("q", [i for i in ii if st[i] not in (a, fr, "normal")], L)
        mg = T.mgrp                                           # M 分组沿用 stage 20c（m_group），只把 repair 拆成 a 与 fr
        spec[f"M_{L}_normal"] = ("m", list(np.flatnonzero(mg == f"M_{L}_normal")))
        spec[f"M_{a}"] = ("m", [i for i in np.flatnonzero(mg == f"M_{L}_repair") if st[i] == a])
        spec[f"M_{fr}"] = ("m", [i for i in np.flatnonzero(mg == f"M_{L}_repair") if st[i] == fr])
        spec[f"M_{L}_other_alt"] = ("m", list(np.flatnonzero(mg == f"M_{L}_other_alt")))
    spec["q_rest"] = ("qrest", [i for i, l in enumerate(T.lin) if l not in ("PT", "TAL")])
    spec["M_rest"] = ("m", list(np.flatnonzero(T.mgrp == "M_rest")))
    return {k: v for k, v in spec.items() if v[0] == "p" or len(v[1])}


class ShapleyGPU:
    def __init__(self, T: FineTissue, spec: dict, device: str):
        self.T, self.spec, self.F = T, spec, list(spec)
        self.dev = torch.device(device)
        self.st = fine_state(T)
        self.normal = {L: [i for i, (l, s) in enumerate(zip(T.lin, self.st)) if l == L and s == "normal"]
                       for L in ("PT", "TAL")}
        q = T.q.astype(np.float64)
        self.q_don, self.q_ref = q, T.qbar.astype(np.float64)
        self.mix_don, self.mix_ref = {}, {}
        for L, ii in self.normal.items():
            tot = q[:, ii].sum(1, keepdims=True)
            self.mix_don[L] = np.where(tot > 0, q[:, ii] / np.where(tot > 0, tot, 1), 0.0)
            rt = self.q_ref[ii].sum()
            self.mix_ref[L] = self.q_ref[ii] / rt if rt > 0 else np.zeros(len(ii))
        self.lin_absent = {L: (T.p[:, T.lins.index(L)] == 0) if L in T.lins else None for L in ("PT", "TAL")}
        self.Mfill = torch.tensor(T.Mfill, device=self.dev)
        self.mbar = torch.tensor(T.mbar, device=self.dev)
        self.Lmask = T.Lmask.astype(np.float64)
        self.n_clipped = 0

    def weights(self, S: set[str]) -> tuple[np.ndarray, np.ndarray]:
        T, nd = self.T, len(self.T.donors)
        p = T.p if "p" in S else np.repeat(T.pbar[None], nd, 0)
        q = np.repeat(self.q_ref[None], nd, 0)
        for name, sp in self.spec.items():
            if sp[0] in ("q", "qrest") and name in S:
                q[:, sp[1]] = self.q_don[:, sp[1]]
        for L, ii in self.normal.items():
            if not ii:
                continue
            owner = next((n for n, sp in self.spec.items() if sp[0] == "q" and sp[2] == L), None)
            lin_i = [i for i, l in enumerate(T.lin) if l == L and i not in ii]
            tot = 1 - q[:, lin_i].sum(1, keepdims=True)
            self.n_clipped += int((tot < 0).sum())
            tot = np.clip(tot, 0, None)
            mix = self.mix_don[L] if owner in S else np.repeat(self.mix_ref[L][None], nd, 0)
            q[:, ii] = tot * mix
            ab = self.lin_absent[L]
            if ab is not None and ab.any():                  # 供体无该谱系细胞：与 stage 20c 同样把该谱系份额置零（仅当 q 因子取供体值）
                on = [n for n, sp in self.spec.items() if sp[0] == "q" and n in S and set(sp[1]) & set(lin_i + ii)]
                if on:
                    q[np.ix_(ab, lin_i + ii)] = self.q_don[np.ix_(ab, lin_i + ii)]
        W = (p @ self.Lmask.T) * q
        md = np.zeros(len(T.fines), bool)
        for name, sp in self.spec.items():
            if sp[0] == "m" and name in S:
                md[sp[1]] = True
        return W, md

    def scores(self, up: list[str], dn: list[str], batch: int) -> np.ndarray:
        n = len(self.F)
        gu = torch.tensor(self.T.genes.get_indexer(up), device=self.dev)
        gd = torch.tensor(self.T.genes.get_indexer(dn), device=self.dev)
        gu, gd = gu[gu >= 0], gd[gd >= 0]
        ng = len(self.T.genes)
        V = np.zeros((1 << n, len(self.T.donors)))
        for start in range(0, 1 << n, batch):
            masks = range(start, min(start + batch, 1 << n))
            Ws, Ms = [], []
            for mk in masks:
                W, md = self.weights({self.F[i] for i in range(n) if mk >> i & 1})
                Ws.append(W), Ms.append(md)
            W = torch.tensor(np.stack(Ws), dtype=torch.float32, device=self.dev)
            md = torch.tensor(np.stack(Ms), dtype=torch.float32, device=self.dev)[:, None, :]
            E = torch.einsum("bdf,dfg->bdg", W * md, self.Mfill) + torch.einsum("bdf,fg->bdg", W * (1 - md), self.mbar)
            srt = torch.sort(E, dim=-1).values.contiguous()

            def mean_rank(idx):
                x = E[..., idx].contiguous()
                lo = torch.searchsorted(srt, x, right=False)
                hi = torch.searchsorted(srt, x, right=True)
                return ((lo + hi + 1).double() / 2 / ng).mean(-1)
            V[start:start + len(masks)] = (mean_rank(gu) - mean_rank(gd)).cpu().numpy()
        return V

    def shapley(self, up: list[str], dn: list[str], batch: int) -> pd.DataFrame:
        n = len(self.F)
        V = self.scores(up, dn, batch)
        full, allm = (1 << n) - 1, np.arange(1 << n)
        pc = np.array([bin(m).count("1") for m in allm])
        w = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
        phi = np.zeros((V.shape[1], n))
        for i in range(n):
            sel = allm[(allm >> i & 1) == 0]
            phi[:, i] = (w[pc[sel]][:, None] * (V[sel | (1 << i)] - V[sel])).sum(0)
        out = pd.DataFrame(phi, index=self.T.donors, columns=self.F)
        ip = self.F.index("p")
        out["full_minus_ref"] = V[full] - V[0]
        out["state_only_minus_ref"] = V[full & ~(1 << ip)] - V[0]
        out["comp_only_minus_ref"] = V[1 << ip] - V[0]
        out["full_score"] = V[full]
        return out
