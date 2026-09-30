"""KPMP 全组织 pseudobulk 上复刻 stage 13 的「DKD vs 其他 CKD」与共享轴调整，并做细胞类型内的标志表达检验。

- 每个供体的全部细胞求和 → log2CPM → 样本内百分位秩（与 bulk 的 full_ranks 同尺度）。
- 共享轴 = stage 13 bulk 的 SHARED 效应（非 DKD 病人 vs 对照，ERCB 等队列），与 KPMP 完全独立；
  得分 = top-k 上调基因平均秩 − top-k 下调基因平均秩（同 lib/injury.injury_score）。
- 调整：在病人（DKD ∪ 其他 CKD）内对每个基因回归掉得分（不用诊断标签），再算 Hedges g。
  残差化不依赖标签，因此对残差矩阵做供体标签置换是有效的零分布。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from lib.kpmp_stats import abundance_test, direction_test, enrichment, hedges_g, label_perms, log2cpm


def tissue_ranks(pb, donors: pd.Index, min_detect_frac: float) -> pd.DataFrame:
    rows, X = pb.aggregate({"ALL": list(pb.rows.fine.unique())})
    rows = rows.set_index("donor")
    idx = rows.index.get_indexer(donors)
    C = X[idx].toarray()
    keep = (C > 0).mean(0) >= min_detect_frac
    L = log2cpm(C[:, keep])
    R = (rankdata(L, axis=1) - 1) / (L.shape[1] - 1)
    return pd.DataFrame(R, index=donors, columns=pb.genes[keep])


def shared_score(R: pd.DataFrame, axis: pd.Series, k: int) -> pd.Series:
    ax = axis.reindex(R.columns).dropna()
    return R[ax.nlargest(k).index].mean(1) - R[ax.nsmallest(k).index].mean(1)


def residualize(R: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    Z = np.c_[np.ones(len(z)), z.values]
    beta = np.linalg.lstsq(Z, R.values, rcond=None)[0]
    return pd.DataFrame(R.values - Z @ beta, index=R.index, columns=R.columns)


def _perm_spearman(M: np.ndarray, n_a: int, bulk: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    assign = label_perms(M.shape[0], n_a, n_perm, rng)
    rb = rankdata(bulk)
    rb = (rb - rb.mean()) / rb.std()
    out = []
    for r in assign:
        g = hedges_g(M[r], M[~r])
        rg = rankdata(g)
        out.append(float(((rg - rg.mean()) / rg.std() * rb).mean()))
    out = np.array(out)
    return out[0], float((1 + np.sum(out[1:] >= out[0])) / len(out))


def tissue_replication(pb, meta: pd.DataFrame, marker_sets: dict, K: dict, n_perm: int, seed: int, rng):
    T = K["tissue"]
    a, b = T["patients"]
    da, db = pb.group_donors(a), pb.group_donors(b)
    donors = da.append(db)
    R = tissue_ranks(pb, donors, T["min_detect_frac"])
    n_a = len(da)
    effects, scores, rep = [], [], []
    variants = [("none", "raw", R)]                          # 未调整：与共享轴无关，只算一次
    for ac in T["axis_compartments"]:
        axis = meta[(meta.compartment == ac) & (meta.contrast == "SHARED")].set_index("gene").g_mean
        z = shared_score(R, axis, T["axis_top_k"])
        scores.append(pd.DataFrame({"dataset": pb.name, "axis": ac, "donor": donors,
                                    "group": [a] * n_a + [b] * len(db), "shared_score": z.values}))
        rep.append({"dataset": pb.name, "axis": ac, "test": f"shared_score_{a}_vs_{b}", "n_A": n_a, "n_B": len(db)}
                   | abundance_test(z.values[:n_a], z.values[n_a:], K["n_boot"], n_perm, rng))
        variants.append((ac, "adj", residualize(R, z)))
    for ac, tag, M in variants:
        g = hedges_g(M.values[:n_a], M.values[n_a:])
        con = f"KPMP_{a}_vs_{b}" + ("_adj" if tag == "adj" else "")
        effects.append(pd.DataFrame({"compartment": ac, "contrast": con, "unit": pb.name, "gene": M.columns, "g": g}))
        for comp in ("GLOM", "TUB"):
            for bc in ("DKD_vs_PAT_adj", "DKD_vs_PAT", "DKD_vs_CTRL", "SHARED"):
                mm = meta[(meta.compartment == comp) & (meta.contrast == bc)].set_index("gene")
                m = mm.g_mean
                gi = M.columns.intersection(m.index)
                rho, p = _perm_spearman(M[gi].values, n_a, m[gi].values, n_perm, rng)
                top = m[mm.sign_consistency == 1]
                top = top[top.abs().nlargest(K["n_top_validate"]).index]
                # 秩尺度上的差值很小，不设 |差值| 下限（lfc_min = 0）
                d = direction_test(M.values, M.columns, n_a, top, m, 0.0, n_perm, rng) or {}
                rep.append({"dataset": pb.name, "axis": ac, "test": f"{con}~bulk_{comp}_{bc}", "n_A": n_a,
                            "n_B": len(db), "n_genes": len(gi), "spearman_all": rho, "p_perm_spearman": p}
                           | {k: d.get(k) for k in ("agree_top", "agree_background", "delta", "p_perm", "rho", "p_rho")})
    E = pd.concat(effects)
    empty = pd.DataFrame(columns=["compartment", "contrast", "gene", "g_mean"])
    enr = pd.concat([enrichment(E, empty, mk, n_perm, seed).assign(reference=ref) for ref, mk in marker_sets.items()])
    enr = enr[enr.unit != "META"].assign(dataset=pb.name)
    return E.assign(dataset=pb.name), enr, pd.DataFrame(rep), pd.concat(scores)


def marker_expression(pb, per_contrast: dict, markers: dict, K: dict, n_perm: int, rng) -> list[dict]:
    """per_contrast：(A, B) → {细胞类型: (rows, L)}（expressed() 的输出）。
    细胞类型内的标志得分（该类 KPMP 标志的平均 log2CPM）在供体水平比较 → 细胞内在表达变化，区别于比例变化。"""
    out = []
    for (a, b), per in per_contrast.items():
        da, db = pb.group_donors(a), pb.group_donors(b)
        for ct, (r, L) in per.items():
            if ct not in markers:
                continue
            gi = pb.genes.get_indexer(markers[ct])
            gi = gi[gi >= 0]
            sa, sb = L[r.donor.isin(da).values][:, gi].mean(1), L[r.donor.isin(db).values][:, gi].mean(1)
            if len(sa) < K["min_donors"] or len(sb) < K["min_donors"]:
                continue
            out.append({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "cell_type": ct, "n_markers": len(gi),
                        "n_A": len(sa), "n_B": len(sb), "mean_A": sa.mean(), "mean_B": sb.mean(),
                        "delta_log2cpm": sa.mean() - sb.mean()} | abundance_test(sa, sb, K["n_boot"], n_perm, rng))
    return out


def old_marker_profile(z: pd.DataFrame, old_markers: dict, name: str) -> pd.DataFrame:
    """旧参照每类标志在 KPMP 参照中的平均特异性 z（旧标志在 KPMP 里对应哪种细胞）。"""
    rows = []
    for c, mk in old_markers.items():
        gi = z.index.intersection(mk)
        if len(gi) < 5:
            continue
        mz = z.loc[gi].mean()
        rows += [{"reference": name, "old_cell_type": c, "kpmp_cell_type": k, "n_genes": len(gi), "mean_z": v,
                  "rank": int(i + 1)} for i, (k, v) in enumerate(mz.sort_values(ascending=False).items())]
    return pd.DataFrame(rows)
