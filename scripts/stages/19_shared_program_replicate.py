"""Stage 19d：SCP 在独立数据中的复现与临床关联（这些数据未参与任何选择；每个检验只运行一次）。

分数 = SCP 带符号平均秩（上调核心 − 下调核心，样本内百分位秩，基因 = 该队列全部样本都测到的基因）。
基因集：full（全部核心）、no_procurement（去掉 KPMP 健康供体「手术 vs 经皮」取材敏感的核心基因）、
        no_ieg（去掉经典即早基因）；另报告另一区室核心作敏感性。
零分布：同大小、同上 / 下调数的随机基因集（n_random_sets 个），报告观测 AUROC / Spearman 的百分位。
临床：GSE175759 eGFR（剔除作者标注的技术离群）；GSE166239 eGFR / 蛋白尿；KPMP 切片 bulk eGFR 分箱与蛋白尿分级；
      GSE142025 晚期 vs 早期 DN；GSE115857 IgAN 分级 G1–G3（GEO 原始字段，分级体系未注明）。
      KPMP snRNA 供体 eGFR 见 kpmp/egfr.tsv。
产出 results/19_shared_program/replicate/
"""
from __future__ import annotations

import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import full_ranks  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp import auc_vs_null, random_set_null, rho_vs_null, signed_score  # noqa: E402
from lib.scp_core import OUT, SP, load_core_table  # noqa: E402
from lib.stats import auc_rows, ci  # noqa: E402

CFG = load_config()
RP = SP["replicate"]
O = OUT / "replicate"
INTERIM = ROOT / CFG["paths"]["interim"]


def gene_sets() -> dict[tuple[str, str], tuple[list, list]]:
    core = load_core_table()
    pr = pd.read_csv(OUT / "kpmp" / "core_procurement.tsv", sep="\t")
    sens = set(pr.gene[pr.procurement_sensitive.astype(str) == "True"] + "|" + pr.compartment[pr.procurement_sensitive.astype(str) == "True"])
    out = {}
    for comp, c in core.groupby("compartment"):
        keys = {"full": c, "no_procurement": c[~(c.gene + "|" + comp).isin(sens)], "no_ieg": c[~c.canonical_ieg]}
        for k, t in keys.items():
            out[(comp, k)] = (t.gene[t.direction == "up"].tolist(), t.gene[t.direction == "down"].tolist())
    return out


def load_cohort(name: str, spec: dict, s, units) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (R 样本 × 基因, meta: diagnosis + 临床列)。"""
    if spec["source"] == "indep":
        ts = pd.read_csv(ROOT / "results/17_independent/test_samples.tsv", sep="\t", index_col=0, low_memory=False)
        ts = ts[ts.cohort == name]
        e = pd.read_parquet(INTERIM / "indep" / f"{name}_expr.parquet")[ts.index]
        e = e.loc[e.notna().all(axis=1)]
        R = ((e.rank(axis=0) - 1) / (e.shape[0] - 1)).T
        return R, ts.loc[R.index]
    idx = units[name]
    m = pd.read_csv(INTERIM / f"{name}_meta.tsv", sep="\t", dtype=str)
    m.index = name + "|" + m.gsm
    ch = m.filter(like="characteristics").fillna("").agg(" ; ".join, axis=1)
    meta = pd.DataFrame(index=m.index)
    meta["egfr"] = ch.str.extract(r"estimated gfr[^:]*:\s*([\d.]+)")[0].astype(float)
    meta["outlier"] = ch.str.contains(r"technical outlier:\s*(?:yes|outlier)", case=False)
    meta["group_raw"] = ch.str.extract(r"group:\s*([\w_]+)")[0]
    meta["grade"] = ch.str.extract(r"classification:\s*G(\d)")[0].astype(float)
    idx = idx[~meta.outlier.reindex(idx).fillna(False).to_numpy(bool)]
    R = full_ranks(idx, s).dropna(axis=1)
    meta = meta.reindex(R.index)
    meta["diagnosis"] = s.loc[R.index, "diagnosis"].to_numpy()
    return R, meta


def boot_ci(y, x, n, rng) -> tuple[float, float]:
    b = []
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n):
        j = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        b.append(auc_rows(y[j], x[j][None])[0])
    return ci(np.array(b))


def clinical(name: str, meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(检验名, 样本掩码, 变量) 列表；变量越大 = 越严重的方向在报告时注明。"""
    d = meta.diagnosis.to_numpy()
    pat = d != "CONTROL"
    out = []
    if name == "GSE175759-RNAseq":
        e = meta.egfr.to_numpy(float)
        out += [("egfr_patients", pat & np.isfinite(e), e), ("egfr_all", np.isfinite(e), e)]
    if name == "GSE166239":
        e = pd.to_numeric(meta["egfr (ml/min/m2)"], errors="coerce").to_numpy()
        p = pd.to_numeric(meta["proteinuria (g/d)"], errors="coerce").to_numpy()
        out += [("egfr_patients", pat & np.isfinite(e), e), ("egfr_all", np.isfinite(e), e),
                ("proteinuria_patients", pat & np.isfinite(p), p), ("proteinuria_all", np.isfinite(p), p)]
    if name == "KPMP":
        e = meta.baseline_egfr.map(lambda v: (int(m.group(1)) + int(m.group(2)) + 1) / 2
                                   if isinstance(v, str) and (m := re.match(r"(\d+)-(\d+)", v)) else np.nan).to_numpy(float)
        pmap = {"<150 mg/g cr (prot)": 0, "150 to <500 mg/g cr (prot)": 1, "500 to <1000 mg/g cr (prot)": 2,
                ">=1000 mg/g cr (prot)": 3}
        p = meta.proteinuria.map(pmap).to_numpy(float)
        out += [("egfr_bin_patients", pat & np.isfinite(e), e), ("proteinuria_cat_patients", pat & np.isfinite(p), p)]
    if name == "GSE142025-RNAseq":
        g = meta.group_raw.map({"Early_DN": 0.0, "Advanced_DN": 1.0}).to_numpy(float)
        out += [("advanced_vs_early_DN", np.isfinite(g), g)]
    if name == "GSE115857-GPL14951":
        g = meta.grade.to_numpy(float)
        out += [("iga_grade_G1_G3", (d == "IgAN") & np.isfinite(g), g)]
    return out


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, units = import_module("11_injury_axis").load_samples()
    sets = gene_sets()
    cohorts = dict(RP["cohorts"])
    for u in RP["also_dkd_control"]:
        cohorts[u] = {"source": "interim", "core": "GLOM" if s.loc[units[u], "compartment"].iloc[0] == "GLOM" else "TUB",
                      "positive": ["DKD"], "supplementary": True}
    aucs, clin, scores = [], [], []
    for name, spec in cohorts.items():
        assert not any(name.startswith(x) or x.startswith(name) for comp in SP["sources"] for x in SP["sources"][comp])
        R, meta = load_cohort(name, spec, s, units)
        d = meta.diagnosis.to_numpy()
        m = np.isin(d, spec["positive"] + ["CONTROL"])
        y = np.isin(d[m], spec["positive"]).astype(int)
        for (comp, kind), (up, dn) in sets.items():
            up_, dn_ = [g for g in up if g in R.columns], [g for g in dn if g in R.columns]
            sc = signed_score(R, up_, dn_).to_numpy()
            primary = comp == spec["core"]
            if kind == "full":
                scores.append(pd.DataFrame({"cohort": name, "core": comp, "sample_uid": R.index, "diagnosis": d,
                                            "scp": sc}))
            null = random_set_null(R, len(up_), len(dn_), RP["n_random_sets"], rng)
            if y.sum() and (1 - y).sum():
                lo, hi = boot_ci(y, sc[m], RP["n_boot"], rng)
                aucs.append({"cohort": name, "core": comp, "gene_set": kind, "primary": primary,
                             "supplementary": bool(spec.get("supplementary", False)),
                             "positive": "+".join(spec["positive"]), "n_pos": int(y.sum()), "n_ctrl": int((1 - y).sum()),
                             "n_up": len(up_), "n_down": len(dn_), **auc_vs_null(y, sc[m], null[:, m]),
                             "auroc_lo": lo, "auroc_hi": hi})
            for test, mk, x in clinical(name, meta):
                if mk.sum() < 5:
                    continue
                row = {"cohort": name, "core": comp, "gene_set": kind, "primary": primary, "test": test,
                       "n": int(mk.sum())}
                if test == "advanced_vs_early_DN":
                    yy = x[mk].astype(int)
                    row |= {"n_advanced": int(yy.sum()), "n_early": int((1 - yy).sum()),
                            **{f"{k}": v for k, v in auc_vs_null(yy, sc[mk], null[:, mk]).items()}}
                else:
                    row |= rho_vs_null(x[mk], sc[mk], null[:, mk])
                clin.append(row)
        print(name, "done", flush=True)
    A, C, S = pd.DataFrame(aucs), pd.DataFrame(clin), pd.concat(scores)
    A.to_csv(O / "auroc.tsv", sep="\t", index=False)
    C.to_csv(O / "clinical.tsv", sep="\t", index=False)
    S.to_csv(O / "scores.tsv", sep="\t", index=False)
    write_provenance("19_shared_program/replicate",
                     [OUT / "core/core_genes.tsv", OUT / "kpmp/core_procurement.tsv",
                      ROOT / "results/17_independent/test_samples.tsv"]
                     + sorted((INTERIM / "indep").glob("*_expr.parquet")),
                     [O / "auroc.tsv", O / "clinical.tsv", O / "scores.tsv"], CFG["seed"],
                     {"n_random_sets": RP["n_random_sets"], "n_boot": RP["n_boot"]})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(A.round(3).to_string())
        print(C.round(3).to_string())


if __name__ == "__main__":
    main()
