"""Stage 16 特征：由 BayesPrism 输出构建样本级特征矩阵（sample_uid 索引；每个区室一套，列随参照类型而定）。

- COMP        类型层 θ（final）的 CLR 变换：log(θ + eps) − 样本内均值。
- SF          状态份额：类型内各状态 θ_state / Σ θ_state（first），logit(· + eps)；单状态类型不产生列。
- RESID       组成残差：bulk 样本内秩 − 组成预测表达（Σ θ_k φ_k）的样本内秩（基因宇宙）。
- Z           细胞内状态：每个类型 k 的 Z_k（BayesPrism 细胞类型特异表达），只取该类型「拥有」的基因
              （参照中 k 的 CPM 占全部类型 CPM 之和 ≥ z_min_share，与 bulk 无关），
              Z_k 按类型总量归一化（去掉 θ_k）→ log2 CPM → 样本内在这些基因上的百分位秩。
              期望意义下 log Z_kg = log X_g + log φ_kg − log Σ_k' θ_k' φ_k'g + 常数，
              因此样本间变化 = bulk 相对「组成预测表达」的残差 —— 正是组成之外的细胞内信号。
全部无监督、不读诊断；芯片平台未测到的基因为 NaN（fold 内用训练集中位数填补，与 stage 15 同）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.deconv_ref import DC, OUT, ref_dir
from lib.repro import ROOT

RUN = ROOT / "results" / "16_deconv" / "run"
FE = DC["features"]


def _comp_ref(comp: str) -> str:
    return DC["ref_compartment"][comp]


@lru_cache(maxsize=None)
def theta_tables(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(RUN / tag / "theta_type.tsv", sep="\t")
    s = pd.read_csv(RUN / tag / "theta_state.tsv", sep="\t")
    return t, s


def comp_features(ref: str, comp_ref: str, tag: str) -> pd.DataFrame:
    t, _ = theta_tables(tag)
    t = t[(t.reference == ref) & (t.compartment.map(_comp_ref) == comp_ref)]
    W = t.pivot_table(index="sample_uid", columns="type", values="theta")
    L = np.log(W + FE["eps"])
    return L.sub(L.mean(axis=1), axis=0).add_prefix("comp:")


def statefrac_features(ref: str, comp_ref: str, tag: str) -> pd.DataFrame:
    _, s = theta_tables(tag)
    s = s[(s.reference == ref) & (s.compartment.map(_comp_ref) == comp_ref)]
    n_states = s.groupby("type").state.nunique()
    s = s[s.type.isin(n_states.index[n_states > 1])].copy()
    tot = s.groupby(["sample_uid", "type"]).theta.transform("sum")
    s["share"] = s.theta / (tot + 1e-12)
    W = s.pivot_table(index="sample_uid", columns="state", values="share")
    W = W.clip(FE["eps"], 1 - FE["eps"])
    return np.log(W / (1 - W)).add_prefix("sf:")


@lru_cache(maxsize=None)
def type_profiles(ref: str, comp_ref: str) -> tuple[list[str], pd.Index, np.ndarray]:
    """参照类型平均谱：类型内全部供体 × 状态行计数之和，按全部参照基因归一化（类型 × 基因）。"""
    d = ref_dir(ref, comp_ref)
    tr = pd.read_parquet(d / "counts_triplet.parquet")
    rows = pd.read_csv(d / "rows.tsv", sep="\t", keep_default_na=False)
    genes = pd.Index((d / "genes.txt").read_text().split("\n")[:-1])
    X = sp.csr_matrix((tr.x, (tr.i - 1, tr.j - 1)), shape=(len(rows), len(genes)))
    types = sorted(rows.type.unique())
    G = sp.csr_matrix((np.ones(len(rows)), (pd.Index(types).get_indexer(rows.type), np.arange(len(rows)))),
                      shape=(len(types), len(rows)))
    P = np.asarray((G @ X).todense())
    return types, genes, P / P.sum(1, keepdims=True)


@lru_cache(maxsize=None)
def owned_genes(ref: str, comp_ref: str) -> dict[str, pd.Index]:
    """参照中每个类型「拥有」的基因：类型 CPM / 全部类型 CPM 之和 ≥ z_min_share。"""
    types, genes, P = type_profiles(ref, comp_ref)
    share = P / (P.sum(0, keepdims=True) + 1e-300)
    return {t: genes[share[i] >= FE["z_min_share"]] for i, t in enumerate(types)}


def resid_features(ref: str, comp_ref: str, tag: str, rank: pd.DataFrame) -> pd.DataFrame:
    """组成残差：样本内秩(bulk) − 样本内秩(组成预测表达 Σ_k θ_k φ_k)，在基因宇宙中该样本测到的基因上计算。
    参照谱 φ 未经平台更新，逐基因的平台常数偏移在同一平台内对所有样本相同。"""
    t, _ = theta_tables(tag)
    t = t[(t.reference == ref) & (t.compartment.map(_comp_ref) == comp_ref)]
    W = t.pivot_table(index="sample_uid", columns="type", values="theta")
    types, genes, P = type_profiles(ref, comp_ref)
    g = rank.columns.intersection(genes)
    idx = W.index.intersection(rank.index)
    Xhat = pd.DataFrame(W.loc[idx, types].to_numpy() @ P[:, genes.get_indexer(g)], index=idx, columns=g)
    R = rank.loc[idx, g]
    Xhat = Xhat.where(R.notna())
    out = R.rank(axis=1, pct=True) - Xhat.rank(axis=1, pct=True)
    return out.astype("float32").add_prefix("resid:")


def _z_batch(out, own: dict[str, pd.Index]) -> pd.DataFrame:
    parts = []
    for t, gk in own.items():
        f = out / f"Z_{t}.parquet"
        if not f.exists():
            continue
        Z = pd.read_parquet(f).set_index("sample_uid")
        tot = Z.sum(axis=1)
        g = Z.columns.intersection(gk)
        if len(g) < FE["z_min_genes"]:
            continue
        L = np.log2(Z[g].div(tot + 1e-12, axis=0) * 1e6 + 1.0)
        R = L.rank(axis=1, pct=True)
        R.columns = [f"z:{t}:{x}" for x in g]
        parts.append(R)
    return pd.concat(parts, axis=1)


def z_features(ref: str, comp_ref: str, tag: str) -> pd.DataFrame:
    man = pd.read_csv(ROOT / "results/16_deconv/prepare/mixture_manifest.tsv", sep="\t")
    man = man[man.reference_compartment == comp_ref]
    own = owned_genes(ref, comp_ref)
    parts = [_z_batch(OUT / "bp" / ref / f"{r.series}__{r.compartment}", own) for r in man.itertuples()
             if (OUT / "bp" / ref / f"{r.series}__{r.compartment}" / "DONE").exists()]
    return pd.concat(parts, axis=0).astype("float32")


def feature_sets(comp: str, tag: str, ref: str, rank: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """区室 comp 的全部特征块。RANK = 组织平均秩（stage 15 特征）；RANK_Z = RANK 限于 Z 块覆盖的基因
    （同基因集对照：区分「细胞内状态表示」与「标志基因集更小」两种解释）；RESID = 组成残差秩。"""
    cr = _comp_ref(comp)
    C, S, Z = comp_features(ref, cr, tag), statefrac_features(ref, cr, tag), z_features(ref, cr, tag)
    zg = pd.Index(sorted({c.split(":", 2)[2] for c in Z.columns}))
    return {"COMP": C, "SF": S, "Z": Z, "RANK": rank, "RANK_Z": rank[rank.columns.intersection(zg)],
            "RESID": resid_features(ref, cr, tag, rank)}

