"""Stage 17b：DKD 关键基因在三个独立队列中的复现（方案见 results/17_independent/PLAN_genes.json，只运行一次）。

分数 = 带符号的样本内百分位秩均值（在该样本全部测到的基因上排序）。
零分布：同大小、同上/下调数量的随机基因集；参照：177 条已发表签名按 stage 12 同法定向后同样打分。
产出 results/17b_key_genes/replication.tsv、published_in_independent.tsv
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "17b_key_genes"
SEX = {"XIST", "TSIX", "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY", "NLGN4Y", "TXLNGY"}
MULTI = {"GLOM": ["COLEC12", "LUM", "ASPN", "CHGA", "ADH1B", "ITIH2", "PLA2G4A", "C1orf115"],
         "TUB": ["ACKR1", "ADH1C", "AGR2", "VCAN", "NPHS2", "CLIC5", "LPL", "RRM2"]}
COHORT_COMP = {"GSE162830": "GLOM", "KPMP": "TUB", "GSE166239": "TUB"}
N_RANDOM = 1000


def cohort_ranks(c: str, samples: pd.DataFrame) -> pd.DataFrame:
    e = pd.read_parquet(ROOT / "data/interim/indep" / f"{c}_expr.parquet")
    e = e[[u for u in samples.sample_uid if u in e.columns]]
    e = e.loc[e.notna().all(axis=1)]
    return ((e.rank(axis=0) - 1) / (e.shape[0] - 1)).T      # 样本 × 基因


def key_sets() -> dict[tuple[str, str], dict[str, int]]:
    t = pd.read_csv(ROOT / "results/18_interpret/summary/dkd_key_gene_evidence.tsv", sep="\t")
    out = {}
    for comp, g in t.groupby("compartment"):
        g = g[~g.gene.isin(SEX)]
        sign = dict(zip(g.gene, np.sign(g.e_full).astype(int)))
        out[(comp, "key50")] = sign
        out[(comp, "multi_chain")] = {x: sign[x] for x in MULTI[comp] if x in sign}
    return out


def score(R: pd.DataFrame, genes: dict[str, int]) -> np.ndarray:
    g = [x for x in genes if x in R.columns]
    w = np.array([genes[x] for x in g], float)
    return R[g].values @ w / max(len(g), 1)


def contrasts(diag: pd.Series) -> dict[str, np.ndarray]:
    d = diag.values
    dkd = d == "DKD"
    return {"DKD_vs_otherCKD": np.where(dkd | ~np.isin(d, ["DKD", "CONTROL"]), dkd.astype(float), np.nan),
            "DKD_vs_healthy": np.where(dkd | (d == "CONTROL"), dkd.astype(float), np.nan)}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(y)
    return float(roc_auc_score(y[m], s[m])) if m.sum() and len(np.unique(y[m])) == 2 else np.nan


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CFG["seed"])
    ts = pd.read_csv(ROOT / "results/17_independent/test_samples.tsv", sep="\t")
    sets = key_sets()
    st12 = import_module("12_signature_audit")
    s, units = import_module("11_injury_axis").load_samples()
    orient = st12.orientation(s, units)
    pubs = [x for x in st12.load_signatures(orient) if x["sig_id"].startswith("PMID")]
    rows, prow = [], []
    for c, comp in COHORT_COMP.items():
        smp = ts[ts.cohort == c]
        R = cohort_ranks(c, smp)
        diag = smp.set_index("sample_uid").diagnosis.reindex(R.index)
        cons = contrasts(diag)
        pool = R.columns.values
        pub_scores = {p["sig_id"]: score(R, p["genes"]) for p in pubs if p["compartment"] == comp
                      and sum(g in R.columns for g in p["genes"]) >= 2}
        for (sc_comp, name), genes in sets.items():
            if sc_comp != comp:
                continue
            present = {g: v for g, v in genes.items() if g in R.columns}
            n_up, n = sum(v > 0 for v in present.values()), len(present)
            sc = score(R, present)
            null = []
            for _ in range(N_RANDOM):
                gg = rng.choice(pool, n, replace=False)
                w = np.r_[np.ones(n_up), -np.ones(n - n_up)]
                null.append(R[gg].values @ w / n)
            for con, y in cons.items():
                obs = auc(y, sc)
                nl = np.array([auc(y, z) for z in null])
                pa = np.array([auc(y, z) for z in pub_scores.values()])
                rows.append({"cohort": c, "compartment": comp, "gene_set": name, "n_genes": n, "contrast": con,
                             "n_dkd": int(np.nansum(y)), "n_neg": int(np.sum(y == 0)), "auroc": obs,
                             "random_median": float(np.nanmedian(nl)),
                             "pct_vs_random": float(np.nanmean(nl < obs)),
                             "published_median": float(np.nanmedian(pa)),
                             "pct_vs_published": float(np.nanmean(pa < obs)), "n_published": int(np.isfinite(pa).sum())})
        for pid, z in pub_scores.items():
            for con, y in cons.items():
                prow.append({"cohort": c, "sig_id": pid, "contrast": con, "auroc": auc(y, z)})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "replication.tsv", sep="\t", index=False)
    pd.DataFrame(prow).to_csv(OUT / "published_in_independent.tsv", sep="\t", index=False)
    write_provenance("17b_key_genes", [ROOT / "results/18_interpret/summary/dkd_key_gene_evidence.tsv",
                                       ROOT / "results/17_independent/test_samples.tsv"],
                     [OUT / "replication.tsv", OUT / "published_in_independent.tsv"], CFG["seed"])
    print(rep.round(3).to_string())


if __name__ == "__main__":
    main()
