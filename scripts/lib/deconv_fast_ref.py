"""Stage 16b 参照：KPMP snRNA 供体 × SubclassLevel2 pseudobulk → 区室参照张量、供体划分、基因集、标志。

估计器假设（写进方法）：
- 细胞类型 k 在供体 d 的每细胞表达 μ_dk = Σ_{f∈k} counts_df / Σ_{f∈k} n_df（UMI/核；核 RNA 含量差异保留在 μ 中）。
  模拟时在 fine 状态层级抽细胞（lib/deconv_fast_sim），因此类型内部的状态构成（如 aPT / PT-S1）随样本变化。
- 比例 = 细胞（核）数比例，不是 RNA 比例。
- 供体划分：按 stage 14 的分组（REF / DKD / OTHER_CKD / HKD / AKI）分层随机留出 holdout_frac；
  留出供体只用于比例误差评估，任何模拟训练、标志、签名矩阵都只用训练供体。
- 基因集：区室内全部 model2 bulk 样本无缺失 ∩ 参照最高类型 log2CPM ≥ min_ref_log2cpm ∩ 秩域偏差过滤
  （bulk 各单元平均秩的均值 vs 先验均值组成下模拟组织的平均秩，|差| ≤ domain_max_absdiff；
  只用表达量、不用任何诊断标签，用于去掉核 / 胞质、3' 偏倚等平台系统差异最大的基因）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.kpmp import Pseudobulk
from lib.repro import load_config

CFG = load_config()
DF = CFG["deconv_fast"]


def load_pseudobulk() -> Pseudobulk:
    name = DF["reference"]
    return Pseudobulk(name, CFG["kpmp"]["datasets"][name], CFG["kpmp"])


def donor_split(pb: Pseudobulk, seed: int) -> pd.DataFrame:
    """供体 → group（按 donor_groups 顺序取第一个命中的组）、split（train / holdout）。"""
    rows = []
    for g in DF["donor_groups"]:
        for d in pb.group_donors(g):
            rows.append((d, g))
    t = pd.DataFrame(rows, columns=["donor", "group"]).drop_duplicates("donor").set_index("donor")
    rng = np.random.default_rng(seed)
    t["split"] = "train"
    for g, sub in t.groupby("group"):
        n_ho = int(round(DF["holdout_frac"] * len(sub)))
        ho = rng.choice(sub.index.to_numpy(), n_ho, replace=False)
        t.loc[ho, "split"] = "holdout"
    return t


@dataclass
class CompRef:
    comp: str
    types: list[str]                 # K
    fine: list[str]                  # F（映射到 types 的 fine 状态）
    fine_type: np.ndarray            # F：fine → type 下标
    donors: pd.DataFrame             # index donor：group split
    genes: pd.Index                  # 候选基因（后续 domain 过滤再缩小）
    counts: np.ndarray               # D × F × G：供体 × fine 的 UMI 计数和（float32）
    ncell: np.ndarray                # D × F：细胞数

    def type_cells(self) -> np.ndarray:
        """D × K 细胞数。"""
        out = np.zeros((self.ncell.shape[0], len(self.types)))
        for k in range(len(self.types)):
            out[:, k] = self.ncell[:, self.fine_type == k].sum(1)
        return out

    def type_counts(self, idx: np.ndarray) -> np.ndarray:
        """供体子集 idx 的 D' × K × G 计数和。"""
        C = self.counts[idx]
        return np.stack([C[:, self.fine_type == k].sum(1) for k in range(len(self.types))], 1)

    def signature(self, idx: np.ndarray) -> np.ndarray:
        """G × K 每细胞表达（UMI/核）：供体子集上 Σ counts / Σ 细胞（汇总后求比，稀有类型更稳）。"""
        C = self.type_counts(idx).sum(0)                         # K × G
        n = self.type_cells()[idx].sum(0)                        # K
        return (C / np.maximum(n, 1)[:, None]).T

    def subset_genes(self, genes: pd.Index) -> "CompRef":
        gi = self.genes.get_indexer(genes)
        assert (gi >= 0).all()
        return CompRef(self.comp, self.types, self.fine, self.fine_type, self.donors, pd.Index(genes),
                       self.counts[:, :, gi], self.ncell)


def build_compref(pb: Pseudobulk, comp: str, donors: pd.DataFrame, bulk_genes: pd.Index) -> CompRef:
    """bulk_genes：该区室 bulk 无缺失基因。返回候选基因上的参照（表达过滤已做）。"""
    mapping = DF["types"][comp]
    types = list(mapping)
    present = set(pb.rows.fine)
    fine = [f for t in types for f in mapping[t] if f in present]
    fine_type = np.array([types.index(next(t for t in types if f in mapping[t])) for f in fine])
    genes = pb.genes[pb.genes.isin(bulk_genes)]
    gi = pb.genes.get_indexer(genes)
    r = pb.rows
    r = r[r.donor.isin(donors.index) & r.fine.isin(fine)]
    di = donors.index.get_indexer(r.donor)
    fi = pd.Index(fine).get_indexer(r.fine)
    X = pb.X[r.index.values][:, gi]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    counts = np.zeros((len(donors), len(fine), len(genes)), np.float32)
    counts[di, fi] = X
    ncell = np.zeros((len(donors), len(fine)))
    ncell[di, fi] = r.n_cells.values
    ref = CompRef(comp, types, fine, fine_type, donors, genes, counts, ncell)
    tr = np.flatnonzero(donors.split.values == "train")
    S = ref.signature(tr)
    cpm = S / S.sum(0, keepdims=True) * 1e6
    keep = np.log2(cpm + 1).max(1) >= DF["genes"]["min_ref_log2cpm"]
    return ref.subset_genes(genes[keep])


def prior_mean(ref: CompRef) -> np.ndarray:
    p = DF["prior"][ref.comp]
    if p is None:
        tr = ref.donors.split.values == "train"
        n = ref.type_cells()[tr]
        m = (n / n.sum(1, keepdims=True)).mean(0)
    else:
        m = np.array([p[t] for t in ref.types], float)
    return m / m.sum()


def within_ranks(X: np.ndarray) -> np.ndarray:
    """样本内百分位秩（0–1，平均秩处理并列）。"""
    from scipy.stats import rankdata
    return ((rankdata(X, axis=1) - 1) / (X.shape[1] - 1)).astype(np.float32)


def domain_filter(ref: CompRef, bulk_rank: pd.DataFrame, units: pd.Series) -> tuple[pd.Index, pd.DataFrame]:
    """bulk_rank：样本 × 候选基因（原始样本内秩）。返回保留基因与逐基因诊断表。"""
    B = bulk_rank.reindex(columns=ref.genes)
    Rb = pd.DataFrame(within_ranks(B.to_numpy()), index=B.index, columns=B.columns)
    b_mean = Rb.groupby(units.reindex(Rb.index).values).mean().mean(0)   # 单元平均的均值：大单元不主导
    tr = np.flatnonzero(ref.donors.split.values == "train")
    S = ref.signature(tr)                                                 # G × K
    mix = S @ prior_mean(ref)
    r_ref = within_ranks(mix[None, :])[0]
    diff = b_mean.to_numpy() - r_ref
    keep = np.abs(diff) <= DF["genes"]["domain_max_absdiff"]
    tab = pd.DataFrame({"gene": ref.genes, "bulk_mean_rank": b_mean.to_numpy(), "ref_rank": r_ref,
                        "diff": diff, "keep": keep})
    return ref.genes[keep], tab


def markers(S: np.ndarray, genes: pd.Index, types: list[str], n: int) -> dict[str, pd.Index]:
    """每类 top-n 特异基因：log2(本类 CPM+1) − log2(其余类最大 CPM+1)，要求本类是最高表达类且
    本类 log2CPM ≥ marker_min_log2cpm（避免芯片背景附近、特异但测不到的基因）。"""
    L = np.log2(S / S.sum(0, keepdims=True) * 1e6 + 1)
    out = {}
    for k, t in enumerate(types):
        other = np.delete(L, k, axis=1).max(1)
        spec = L[:, k] - other
        ok = (spec > 0) & (L[:, k] >= DF["genes"]["marker_min_log2cpm"])
        order = np.argsort(-np.where(ok, spec, -np.inf))[:n]
        out[t] = genes[order[ok[order]]]
    return out
