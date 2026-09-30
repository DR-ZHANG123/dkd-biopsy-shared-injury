"""Stage 12：已发表 DKD 签名的损伤匹配评估。

每条签名 = 基因集合 + 方向（文中未给方向时，用定向队列 GSE30528 / GSE30529 的 DKD vs 对照符号定向，
这两个队列因此不参与评估）。样本分数 = 上调基因平均秩 − 下调基因平均秩。
在每个评估单元报告：
  auc_dkd_ctrl        DKD vs 对照
  auc_dkd_other       DKD vs 其他 CKD
  adj_auc_dkd_other   以外部损伤分数（stage 11）为协变量的调整 AUROC
  strat_auc_dkd_other 损伤分层 AUROC（敏感性分析）
  specificity_ratio   d(DKD vs 对照) / d(其他 CKD vs 对照)，≈1 即读出的是非特异损伤
  r_injury            签名分数与损伤分数的 Spearman
并与同大小、同样方式定向的随机基因集（n_random_sets 个）比较，给出经验百分位。
参考签名：INJURY（stage 11 损伤分数，不含 DKD 信息）与 ORIENT_DE_k（定向队列 DKD vs 对照 top-k 基因）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import NON_DISEASE, adjusted_auroc, full_ranks, stratified_auroc  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
CFG = load_config()
OUT = ROOT / "results" / "12_signature_audit"
ORIENT = {"GLOM": "GSE30528-GPL571CEL", "TUB": "GSE30529-GPL571CEL"}
SIG_FILE = ROOT / "metadata" / "published_dkd_signatures.tsv"


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sd + 1e-9))


def metrics(score: np.ndarray, diag: np.ndarray, inj: np.ndarray) -> dict:
    ctrl, dkd = diag == "CONTROL", diag == "DKD"
    oth = ~np.isin(diag, list(NON_DISEASE))
    out = {"auc_dkd_ctrl": np.nan, "auc_dkd_other": np.nan, "adj_auc_dkd_other": np.nan,
           "strat_auc_dkd_other": np.nan, "specificity_ratio": np.nan,
           "r_injury": float(spearmanr(score, inj)[0])}
    if dkd.sum() >= 3 and ctrl.sum() >= 3:
        m = dkd | ctrl
        out["auc_dkd_ctrl"] = roc_auc_score(dkd[m], score[m])
    if dkd.sum() >= 3 and oth.sum() >= 3:
        m = dkd | oth
        y = dkd[m].astype(int)
        out["auc_dkd_other"] = roc_auc_score(y, score[m])
        out["adj_auc_dkd_other"] = adjusted_auroc(y, score[m], inj[m])
        out["strat_auc_dkd_other"] = stratified_auroc(y, score[m], inj[m], CFG["injury"]["n_strata"])
        if ctrl.sum() >= 3:
            d_other = cohen_d(score[oth], score[ctrl])
            out["specificity_ratio"] = cohen_d(score[dkd], score[ctrl]) / d_other if abs(d_other) > 1e-6 else np.nan
    return out


def orientation(s: pd.DataFrame, units: dict) -> dict[str, pd.Series]:
    """定向队列内每个基因 DKD − 对照 的平均秩差。"""
    out = {}
    for comp, u in ORIENT.items():
        idx = units[u]
        R = full_ranks(idx, s).dropna(axis=1)
        d = s.loc[idx, "diagnosis"].values
        out[comp] = R[d == "DKD"].mean() - R[d == "CONTROL"].mean()
    return out


def load_signatures(orient: dict[str, pd.Series]) -> list[dict]:
    sigs = []
    for comp, o in orient.items():
        sigs.append({"sig_id": "INJURY", "compartment": comp, "genes": None})
        for k in (50, 200):
            top = o.abs().nlargest(k).index
            sigs.append({"sig_id": f"ORIENT_DE_{k}", "compartment": comp,
                         "genes": dict(zip(top, np.sign(o[top]).astype(int)))})
    if SIG_FILE.exists():
        tab = pd.read_csv(SIG_FILE, sep="\t", dtype=str)
        for i, r in tab.iterrows():
            genes = [g.strip() for g in str(r["genes"]).split(";") if g.strip()]
            dirs = [d.strip() for d in str(r.get("direction", "NA")).split(";")]
            for comp in ORIENT:
                o = orient[comp]
                gd = {}
                for j, g in enumerate(genes):
                    if g not in o.index:
                        continue
                    dj = dirs[j] if len(dirs) == len(genes) else "NA"
                    gd[g] = 1 if dj == "up" else -1 if dj == "down" else int(np.sign(o[g]) or 1)
                if len(gd) >= 2:
                    sigs.append({"sig_id": f"PMID{r['pmid']}_{i}", "compartment": comp, "genes": gd,
                                 "n_listed": len(genes)})
    return sigs


def score_sig(R: pd.DataFrame, genes: dict) -> np.ndarray:
    g = [x for x in genes if x in R.columns]
    w = np.array([genes[x] for x in g], dtype=float)
    return R[g].values @ w / len(g)


def random_null(R: pd.DataFrame, o: pd.Series, n: int, diag, inj, rng) -> pd.DataFrame:
    pool = R.columns.intersection(o.index).values
    rows = []
    for _ in range(CFG["injury"]["n_random_sets"]):
        g = rng.choice(pool, n, replace=False)
        w = np.sign(o[g].values)
        w[w == 0] = 1
        rows.append(metrics(R[g].values @ w / n, diag, inj))
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts" / "stages"))
    s = pd.read_csv(ROOT / "data/processed/samples.tsv", sep="\t").set_index("sample_uid")
    from importlib import import_module
    units = import_module("11_injury_axis").load_samples()[1]
    inj_all = pd.read_csv(ROOT / "results/11_injury/injury_scores.tsv", sep="\t")
    orient = orientation(s, units)
    sigs = load_signatures(orient)
    rng = np.random.default_rng(CFG["seed"])
    rows, null_cache = [], {}
    for comp, spec in CFG["injury"]["compartments"].items():
        for ev in [u for u in spec["eval"] if u != ORIENT[comp]]:
            ij = inj_all[(inj_all.eval_unit == ev) & (inj_all.compartment == comp)].set_index("sample_uid")
            R = full_ranks(ij.index, s).dropna(axis=1)
            diag, inj = ij["diagnosis"].values, ij["injury"].values
            for sg in [x for x in sigs if x["compartment"] == comp]:
                if sg["genes"] is None:
                    sc, n = inj, np.nan
                else:
                    gd = {g: v for g, v in sg["genes"].items() if g in R.columns}
                    if len(gd) < 2:
                        continue
                    sc, n = score_sig(R, gd), len(gd)
                rec = {"sig_id": sg["sig_id"], "compartment": comp, "eval_unit": ev, "n_genes_used": n,
                       **metrics(sc, diag, inj)}
                if not np.isnan(n):
                    key = (ev, int(min(n, 300)))
                    if key not in null_cache:
                        null_cache[key] = random_null(R, orient[comp], key[1], diag, inj, rng)
                    nl = null_cache[key]
                    for mcol in ("auc_dkd_ctrl", "auc_dkd_other", "adj_auc_dkd_other", "specificity_ratio"):
                        v = rec[mcol]
                        rec[f"{mcol}_null_median"] = float(nl[mcol].median())
                        rec[f"{mcol}_pct_vs_random"] = float((nl[mcol] < v).mean()) if not np.isnan(v) else np.nan
                rows.append(rec)
            print(ev, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "signature_metrics.tsv", sep="\t", index=False)
    write_provenance("12_signature_audit", [ROOT / "results/11_injury/injury_scores.tsv"] +
                     ([SIG_FILE] if SIG_FILE.exists() else []), [OUT / "signature_metrics.tsv"], CFG["seed"],
                     {"n_signatures": int(res.sig_id.nunique())})
    cols = ["sig_id", "eval_unit", "n_genes_used", "auc_dkd_ctrl", "auc_dkd_ctrl_null_median", "auc_dkd_other",
            "adj_auc_dkd_other", "specificity_ratio", "r_injury"]
    print(res[[c for c in cols if c in res]].round(3).to_string())


if __name__ == "__main__":
    main()
