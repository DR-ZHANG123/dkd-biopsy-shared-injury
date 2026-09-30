"""Stage 14 统计：参照标志、富集（同 stage 13 算法）、供体水平差异、方向一致性置换检验、丰度检验。

所有检验以供体为重复单位；置换均为供体标签置换（组大小不变），组合数不超过 n_perm 时精确枚举。
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats


def log2cpm(X: sp.csr_matrix | np.ndarray) -> np.ndarray:
    X = X.toarray() if sp.issparse(X) else np.asarray(X, float)
    lib = X.sum(1, keepdims=True)
    return np.log2(X / np.maximum(lib, 1) * 1e6 + 1)


def hedges_g(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / (sd + 1e-6)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ---------------- 参照标志 ----------------
def _spec_z(L: pd.DataFrame, lab: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_ct = L.groupby(lab.values).mean().T                   # 基因 × 细胞类型
    z = mean_ct.sub(mean_ct.mean(1), axis=0).div(mean_ct.std(1) + 1e-6, axis=0)
    return z, mean_ct


def build_markers(L: pd.DataFrame, rows: pd.DataFrame, cfg: dict,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L：行 = 参照供体 × 细胞类型 的 log2CPM（已按 min_cells 过滤），rows 同序（donor, cell_type）。
    返回 (标志长表, 全体参照供体上的 基因 × 细胞类型 特异性 z)。"""
    n, lab = cfg["n_markers"], rows.cell_type
    z, mean_ct = _spec_z(L, lab)
    donors = rows.donor.unique()
    hits = pd.DataFrame(0, index=L.columns, columns=z.columns)
    n_half = 0
    for _ in range(cfg["marker_resamples"]):
        perm = rng.permutation(donors)
        for half in (perm[: len(perm) // 2], perm[len(perm) // 2:]):
            m = rows.donor.isin(half).values
            zh, mh = _spec_z(L[m], lab[m])
            for c in z.columns:
                if c in zh:
                    top = zh[c][mh[c] >= cfg["min_log2cpm"]].nlargest(2 * n).index
                    hits.loc[top, c] += 1
            n_half += 1
    stab = hits / n_half
    out = []
    nd = rows.groupby("cell_type").donor.nunique()
    for c in z.columns:
        if nd.get(c, 0) < cfg["min_donors"]:
            continue
        ok = (mean_ct[c] >= cfg["min_log2cpm"]) & (stab[c] >= cfg["marker_stability"])
        top = z[c][ok].nlargest(n)
        out += [{"cell_type": c, "gene": g, "z": v, "mean_log2cpm": mean_ct.at[g, c],
                 "stability": stab.at[g, c], "rank": i + 1} for i, (g, v) in enumerate(top.items())]
    return pd.DataFrame(out), z


# ---------------- 富集（同 stage 13） ----------------
def enrichment(E: pd.DataFrame, meta: pd.DataFrame, markers: dict, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    enr = []
    frame = pd.concat([E.assign(level="unit"), meta.rename(columns={"g_mean": "g"}).assign(unit="META")])
    for (comp, con, unit), g in frame.groupby(["compartment", "contrast", "unit"]):
        v = g.set_index("gene").g.dropna()
        for c, mk in markers.items():
            mk = v.index.intersection(mk)
            if len(mk) < 10:
                continue
            obs = v[mk].mean()
            null = np.array([v.values[rng.choice(len(v), len(mk), replace=False)].mean() for _ in range(n_perm)])
            enr.append({"compartment": comp, "contrast": con, "unit": unit, "cell_type": c,
                        "n_markers": len(mk), "mean_g": obs, "z": (obs - null.mean()) / null.std()})
    return pd.DataFrame(enr)


# ---------------- 供体水平差异 ----------------
def welch_de(LA: np.ndarray, LB: np.ndarray) -> dict[str, np.ndarray]:
    t, p = stats.ttest_ind(LA, LB, axis=0, equal_var=False)
    return {"log2fc": LA.mean(0) - LB.mean(0), "t": t, "p": p, "fdr": bh(np.nan_to_num(p, nan=1.0)),
            "g": hedges_g(LA, LB)}


def label_perms(n: int, n_a: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """返回 (P, n) 布尔矩阵，每行 = 一次 A 组指派；第 0 行为观测指派（前 n_a 个为 A）。"""
    obs = np.r_[np.ones(n_a, bool), np.zeros(n - n_a, bool)]
    if comb(n, n_a) <= n_perm:
        rows = []
        for idx in itertools.combinations(range(n), n_a):
            r = np.zeros(n, bool)
            r[list(idx)] = True
            if not (r == obs).all():
                rows.append(r)
        perms = np.array(rows) if rows else np.zeros((0, n), bool)
    else:
        perms = np.array([rng.permutation(obs) for _ in range(n_perm)])
    return np.vstack([obs, perms])


def _lfc_all(L: np.ndarray, assign: np.ndarray) -> np.ndarray:
    A = assign / assign.sum(1, keepdims=True)
    B = ~assign / (~assign).sum(1, keepdims=True)
    return (A - B) @ L                                         # (P, 基因)


def _rank_rows(M: np.ndarray) -> np.ndarray:
    return M.argsort(1).argsort(1).astype(float)


def direction_test(L: np.ndarray, genes: pd.Index, n_a: int, bulk_top: pd.Series, bulk_bg: pd.Series,
                   lfc_min: float, n_perm: int, rng: np.random.Generator) -> dict | None:
    """L：供体 × 基因 log2CPM（前 n_a 行为 A 组）。bulk_top / bulk_bg：基因 → bulk g。"""
    ti = genes.get_indexer(bulk_top.index)
    bi = genes.get_indexer(bulk_bg.index)
    top_g = bulk_top.values[ti >= 0]
    top_s, ti = np.sign(top_g), ti[ti >= 0]
    bg_s, bi = np.sign(bulk_bg.values[bi >= 0]), bi[bi >= 0]
    if len(ti) < 10:
        return None
    assign = label_perms(L.shape[0], n_a, n_perm, rng)
    lt = _lfc_all(L[:, ti], assign)
    lb = _lfc_all(L[:, bi], assign)
    mt, mb = np.abs(lt) > lfc_min, np.abs(lb) > lfc_min
    n_top = mt.sum(1)
    agree_t = np.where(n_top > 0, ((np.sign(lt) == top_s) & mt).sum(1) / np.maximum(n_top, 1), np.nan)
    agree_b = ((np.sign(lb) == bg_s) & mb).sum(1) / np.maximum(mb.sum(1), 1)
    delta = agree_t - agree_b
    rt, rg = _rank_rows(lt), stats.rankdata(top_g)
    rt = rt - rt.mean(1, keepdims=True)
    rg = rg - rg.mean()
    rho = (rt @ rg) / (np.sqrt((rt ** 2).sum(1)) * np.sqrt((rg ** 2).sum()) + 1e-12)
    nperm = len(assign) - 1
    ok = ~np.isnan(delta[1:])
    return {"n_genes_top": int(n_top[0]), "n_genes_top_detected": len(ti), "agree_top": agree_t[0],
            "agree_background": agree_b[0], "delta": delta[0],
            "p_perm": (1 + np.sum(delta[1:][ok] >= delta[0])) / (1 + ok.sum()) if n_top[0] >= 10 else np.nan,
            "perm_mean_agree": np.nanmean(agree_t[1:]), "rho": rho[0],
            "p_rho": (1 + np.sum(rho[1:] >= rho[0])) / (1 + nperm), "n_perm": nperm,
            "exact": nperm < n_perm}


# ---------------- 丰度 ----------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    d = np.sign(a[:, None] - b[None, :])
    return float(d.mean())


def _g(a: np.ndarray, b: np.ndarray) -> float:
    return float(hedges_g(a[:, None], b[:, None])[0])


def abundance_test(a: np.ndarray, b: np.ndarray, n_boot: int, n_perm: int, rng: np.random.Generator) -> dict:
    """a、b：logit 比例（供体）。Hedges g + 组内 bootstrap CI；MWU；Cliff's delta；标签置换 p（双侧）。"""
    g = _g(a, b)
    boots = [_g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    x = np.r_[a, b]
    assign = label_perms(len(x), len(a), n_perm, rng)
    diffs = np.array([_g(x[r], x[~r]) for r in assign[1:]])
    return {"hedges_g": g, "g_ci_lo": float(np.nanpercentile(boots, 2.5)), "g_ci_hi": float(np.nanpercentile(boots, 97.5)),
            "mwu_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
            "cliffs_delta": cliffs_delta(a, b),
            "perm_p": float((1 + np.sum(np.abs(diffs) >= abs(g))) / (1 + len(diffs)))}
