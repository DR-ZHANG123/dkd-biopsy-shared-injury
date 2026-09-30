"""Stage 20 公共工具：KPMP 小管细胞状态（正常 / aPT / frPT / dPT / cycPT；TAL 同）与 SCP 的关系。

- STATES：细粒度名（SubclassLevel2）→ (谱系, 状态)；谱系 / 状态定义只来自 config.repair_state.lineages。
- donor_categories()：供体病理类别（REF / DKD / HKD / OTHER / CKD_unadj / AKI / DM_R），条件取交集，未命中 = NaN。
- state_fractions()：供体 × 状态 占该谱系细胞的比例（含「修复失败」合并 = repair_states 之和）。
- state_pseudobulk()：供体 × 状态 计数（同谱系同状态的细粒度名求和），可选只保留 ≥ min_cells 的行。
- paired_program()：供体内配对（状态 vs 同供体正常细胞）逐基因 log2FC、d_z、配对 t、BH-FDR；不读疾病标签。
- rank_score()：样本内百分位秩上 上调集均值 − 下调集均值（同 SCP 打分形式）。
供体为重复单位；全部置换在供体水平。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from lib.kpmp_stats import label_perms, log2cpm
from lib.repro import ROOT, load_config
from lib.scp import bh

CFG = load_config()
RS = CFG["repair_state"]
K = CFG["kpmp"]
OUT = ROOT / "results" / "20_repair_state"
EGFR = "Baseline eGFR (ml/min/1.73m2) (Binned)"
STATES = {f: (lin, st) for lin, d in RS["lineages"].items() for st, fs in d.items() for f in fs}


def repair_label(lin: str) -> str:
    return "rf" + lin                                       # 例：rfPT = aPT ∪ frPT


def donor_categories(pb, name: str) -> pd.Series:
    cats = RS["categories"][name]
    don = pb.donors
    out = pd.Series(np.nan, index=don.index, dtype=object)
    for cat, cond in cats.items():
        ok = np.ones(len(don), bool)
        for col, vals in cond.items():
            ok &= don[col].astype(str).isin(vals).to_numpy()
        clash = ok & out.notna().to_numpy()
        if clash.any():
            raise ValueError(f"{name}: 供体同时命中多个类别 {cat}")
        out[ok] = cat
    return out.rename("category")


def category_index(cat: pd.Series, group: str) -> pd.Index:
    if group == "CKD":
        return cat.index[cat.isin(RS["ckd"]).to_numpy()]
    return cat.index[(cat == group).to_numpy()]


def state_fractions(pb) -> pd.DataFrame:
    """长表：donor, lineage, state, n_state, n_lineage, frac。state 含各状态与 rf<谱系>（修复失败合并）。"""
    fc = pb.fine_counts
    rows = []
    for lin, d in RS["lineages"].items():
        cols = {st: [f for f in fs if f in fc.columns] for st, fs in d.items()}
        tot = sum(fc[c].sum(1) for c in cols.values() if c)
        per = {st: fc[c].sum(1) if c else pd.Series(0, index=fc.index) for st, c in cols.items()}
        per[repair_label(lin)] = sum(per[s] for s in RS["repair_states"][lin])
        per["altered"] = sum(v for s, v in per.items() if s not in ("normal", repair_label(lin)))
        for st, n in per.items():
            rows.append(pd.DataFrame({"donor": fc.index.astype(str), "lineage": lin, "state": st,
                                      "n_state": n.to_numpy(int), "n_lineage": tot.to_numpy(int)}))
    t = pd.concat(rows, ignore_index=True)
    t["frac"] = t.n_state / t.n_lineage.where(t.n_lineage > 0)
    return t


def state_map(pb, lineage: str | None = None, merge_repair: bool = False) -> dict[str, list[str]]:
    """aggregate() 用的 标签 → 细粒度名 映射；标签 = <谱系>|<状态>。"""
    present = set(pb.rows.fine)
    out: dict[str, list[str]] = {}
    for lin, d in RS["lineages"].items():
        if lineage and lin != lineage:
            continue
        for st, fs in d.items():
            fs = [f for f in fs if f in present]
            if fs:
                out[f"{lin}|{st}"] = fs
        if merge_repair:
            out[f"{lin}|{repair_label(lin)}"] = [f for s in RS["repair_states"][lin] for f in d[s] if f in present]
        out[f"{lin}|all"] = [f for fs in d.values() for f in fs if f in present]
    return out


def state_pseudobulk(pb, min_cells: int) -> dict[str, tuple[pd.Index, sp.csr_matrix, np.ndarray]]:
    """标签 → (供体, 计数矩阵 供体 × 全部基因, 细胞数)。一个细粒度名可同时属于 状态 / rf / all 标签，逐标签聚合。"""
    out = {}
    for lab, fs in state_map(pb, merge_repair=True).items():
        rows, X = pb.aggregate({lab: fs})
        keep = rows.n_cells.to_numpy() >= min_cells
        out[lab] = (pd.Index(rows.donor[keep]), X[np.flatnonzero(keep)], rows.n_cells.to_numpy()[keep])
    return out


def paired_program(pbs: dict, lin: str, state: str, genes: pd.Index, donors: pd.Index | None = None) -> pd.DataFrame:
    """状态 vs 同谱系正常细胞，供体内配对。返回 gene × 统计量（只含两者之一均值 ≥ min_log2cpm 的基因）。"""
    ds, Xs, _ = pbs[f"{lin}|{state}"]
    dn, Xn, _ = pbs[f"{lin}|normal"]
    common = ds.intersection(dn)
    if donors is not None:
        common = common.intersection(donors)
    if len(common) < RS["min_donors"]:
        return pd.DataFrame()
    Ls = log2cpm(Xs[ds.get_indexer(common)])
    Ln = log2cpm(Xn[dn.get_indexer(common)])
    keep = np.maximum(Ls.mean(0), Ln.mean(0)) >= RS["program"]["min_log2cpm"]
    D = Ls[:, keep] - Ln[:, keep]
    m, sd = D.mean(0), D.std(0, ddof=1)
    t = m / (sd / np.sqrt(len(D)) + 1e-12)
    p = 2 * stats.t.sf(np.abs(t), len(D) - 1)
    return pd.DataFrame({"lineage": lin, "state": state, "gene": genes[keep], "n_donors": len(common),
                         "mean_state": Ls[:, keep].mean(0), "mean_normal": Ln[:, keep].mean(0), "log2fc": m,
                         "d_z": m / (sd + 1e-9), "t": t, "p": p, "fdr": bh(p), "frac_donors_up": (D > 0).mean(0)})


def call_program(t: pd.DataFrame) -> pd.DataFrame:
    """标记显著上 / 下调（FDR 与 |lfc|）与打分用 top-n（按 d_z）。"""
    P = RS["program"]
    t = t.copy()
    sig = (t.fdr < P["fdr"]) & (t.log2fc.abs() >= P["lfc"])
    t["call"] = np.where(sig, np.where(t.log2fc > 0, "up", "down"), "")
    t["in_score_set"] = False
    for dr, asc in (("up", False), ("down", True)):
        idx = t[t.call == dr].sort_values("d_z", ascending=asc).index[: P["n_top"]]
        t.loc[idx, "in_score_set"] = True
    return t


def rank_score(E: np.ndarray, genes: pd.Index, up: list[str], dn: list[str]) -> np.ndarray:
    R = stats.rankdata(E, axis=1) / E.shape[1]
    u, d = genes.get_indexer(up), genes.get_indexer(dn)
    u, d = u[u >= 0], d[d >= 0]
    return R[:, u].mean(1) - (R[:, d].mean(1) if len(d) else 0.0)


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    """双侧置换 P（打乱 y）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    if len(x) < 4:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    return float(r), float((1 + (np.abs(null) >= abs(r) - 1e-12).sum()) / (1 + n_perm))


def group_contrast(v: np.ndarray, donors: pd.Index, da: pd.Index, db: pd.Index, n_perm: int, rng) -> dict:
    """v：供体值（donors 同序）；A − B 的均值差、Hedges g、AUROC、双侧标签置换 P。"""
    ia, ib = donors.get_indexer(da.intersection(donors)), donors.get_indexer(db.intersection(donors))
    a, b = v[ia], v[ib]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n_A": len(a), "n_B": len(b)}
    x = np.r_[a, b]
    assign = label_perms(len(x), len(a), n_perm, rng)
    diff = np.array([x[r].mean() - x[~r].mean() for r in assign])
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    auc = float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))
    return {"n_A": len(a), "n_B": len(b), "mean_A": a.mean(), "mean_B": b.mean(), "median_A": np.median(a),
            "median_B": np.median(b), "delta": diff[0], "hedges_g": diff[0] / (sd + 1e-12), "auroc": auc,
            "p_perm": float((1 + (np.abs(diff[1:]) >= abs(diff[0]) - 1e-12).sum()) / len(diff))}


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expr_matched_null(values: pd.Series, expr: pd.Series, members: list[str], n_sets: int, n_bins: int,
                      rng) -> dict:
    """members 的 values 均值 vs 同表达分位箱分布抽取的随机集（n_sets 次）。"""
    g = values.index.intersection(expr.index)
    v, e = values[g], expr[g]
    mem = [m for m in members if m in g]
    if len(mem) < 5:
        return {"n": len(mem)}
    bins = pd.qcut(e.rank(method="first"), n_bins, labels=False)
    pools = {b: np.flatnonzero(bins.to_numpy() == b) for b in range(n_bins)}
    need = bins[mem].value_counts()
    vv = v.to_numpy()
    null = np.empty(n_sets)
    for i in range(n_sets):
        pick = np.concatenate([rng.choice(pools[b], k, replace=False) for b, k in need.items()])
        null[i] = vv[pick].mean()
    obs = float(v[mem].mean())
    return {"n": len(mem), "obs_mean": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "z": float((obs - null.mean()) / (null.std() + 1e-12)),
            "p_upper": float((1 + (null >= obs).sum()) / (1 + n_sets)),
            "p_lower": float((1 + (null <= obs).sum()) / (1 + n_sets))}
