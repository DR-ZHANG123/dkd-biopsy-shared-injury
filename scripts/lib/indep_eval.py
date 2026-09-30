"""Stage 17 用：独立测试队列的样本选择（按 PLAN.json）、样本内秩特征、共享轴分数与配对 bootstrap。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.injury import adjusted_auroc, injury_score
from lib.stats import auc_rows, boot_p_two_sided, ci, cluster_boot_index

# PLAN.json（commit 1ab2135）的逐队列规则：模型区室、阳性、主/次对比的阴性类
COHORTS = {
    "GSE162830": {"model": "GLOM", "compartment": "GLOM", "primary_neg": ["ING"], "secondary_neg": ["CONTROL"],
                  "drop_participants": ["DIA18"]},                     # DIA1 = DIA18（同一患者），只留 DIA1
    "KPMP": {"model": "TUB", "compartment": "WHOLE", "primary_neg": ["HT", "OTHER"], "secondary_neg": ["CONTROL"],
             "drop_participants": []},                                 # 切片 bulk；CKD 未定型不进任何对比
    "GSE166239": {"model": "TUB", "compartment": "WHOLE", "primary_neg": ["HT"], "secondary_neg": ["CONTROL"],
                  "drop_participants": []},
}


def select_test(S: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """每个参与者一个文库：优先 Bulk Total/mRNA 专门测序（strategy = bulk），其次文库读数最多者。"""
    r = COHORTS[cohort]
    keep = r["primary_neg"] + r["secondary_neg"] + ["DKD"]
    t = S[(S.cohort == cohort) & (S.compartment == r["compartment"]) & S.diagnosis.isin(keep)
          & ~S.participant.isin(r["drop_participants"])].copy()
    t["_pref"] = (t.get("strategy", pd.Series("", index=t.index)) == "bulk").astype(int)
    t = t.sort_values(["participant", "_pref", "library_reads"], ascending=[True, False, False])
    return t[~t.participant.duplicated()].drop(columns="_pref")


def cohort_ranks(expr: pd.DataFrame) -> pd.DataFrame:
    """与 lib.injury._series_ranks 相同：只用该队列全部样本都测到的基因，样本内百分位秩（样本 × 基因）。"""
    e = expr.loc[expr.notna().all(axis=1)]
    return ((e.rank(axis=0) - 1) / (e.shape[0] - 1)).T.astype("float32")


def test_arrays(R: pd.DataFrame, genes: list[str], med: np.ndarray) -> np.ndarray:
    """基因宇宙上的秩；该队列未测到的基因用训练集中位数填补（与 m2_data.impute 相同）。"""
    X = R.reindex(columns=genes).to_numpy(np.float32)
    return np.where(np.isnan(X), med[None, :], X)


def test_axis_score(R: pd.DataFrame, axis: pd.Series) -> pd.Series:
    return injury_score(R.dropna(axis=1), axis)


def _adj_rows(y: np.ndarray, S: np.ndarray, a: np.ndarray) -> np.ndarray:
    if y.sum() == 0 or y.sum() == len(y) or np.ptp(a[y == 0]) == 0:
        return np.full(S.shape[0], np.nan)
    return np.array([adjusted_auroc(y, s, a) if np.isfinite(s).all() else np.nan for s in S])


def contrast_metrics(g: pd.DataFrame, methods: list[str], refs: list[str], n_boot: int, seed: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """g：一个 (队列, 对比) 的长表（method sample_uid y score a participant）。患者层面配对 bootstrap。"""
    W = g.pivot(index="sample_uid", columns="method", values="score")[methods]
    meta = g.drop_duplicates("sample_uid").set_index("sample_uid").loc[W.index]
    y, a = meta.y.to_numpy(int), meta.a.to_numpy(float)
    S = W.to_numpy(float).T
    raw, adj = auc_rows(y, S), _adj_rows(y, S, a)
    idx = cluster_boot_index(meta.participant.to_numpy(), n_boot, np.random.default_rng(seed))
    Braw = np.stack([auc_rows(y[i], S[:, i]) for i in idx])
    Badj = np.stack([_adj_rows(y[i], S[:, i], a[i]) for i in idx])
    rows = []
    for k, m in enumerate(methods):
        lo, hi = ci(Braw[:, k])
        alo, ahi = ci(Badj[:, k])
        rows.append({"method": m, "n_dkd": int(y.sum()), "n_neg": int((1 - y).sum()), "auroc": raw[k],
                     "auroc_lo": lo, "auroc_hi": hi, "adj_auroc": adj[k], "adj_lo": alo, "adj_hi": ahi})
    deltas = []
    for ref in refs:
        j = methods.index(ref)
        for k, m in enumerate(methods):
            if m == ref:
                continue
            for kind, pt, B in (("raw", raw, Braw), ("adj", adj, Badj)):
                d = B[:, k] - B[:, j]
                lo, hi = ci(d)
                deltas.append({"method": m, "vs": ref, "metric": kind, "delta": pt[k] - pt[j], "lo": lo, "hi": hi,
                               "p_boot": boot_p_two_sided(d), "n_boot_valid": int(np.isfinite(d).sum())})
    return pd.DataFrame(rows), pd.DataFrame(deltas)
