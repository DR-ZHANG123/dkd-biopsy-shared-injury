"""Stage 09：模型比较。

1. 种子集成：同一 (task, compartment, fold, model, sample) 的 score 取均值；阈值取各种子阈值的均值。
2. 每个 fold（外部测试队列）× 模型：AUROC（患者层面 bootstrap 95% CI）、AUPRC、Brier、校准斜率/截距、
   Youden 阈值下的敏感度/特异度。
3. 配对比较（config compare.pairs，并自动扩展到各 label fraction）：同一组 bootstrap 重抽样索引下的 ΔAUROC。
4. 合并（task×compartment 与 task 两个层级）：DerSimonian–Laird 合并 logit AUROC（方差取 bootstrap）
   与 ΔAUROC，给出预测区间；另报「宏平均 ΔAUROC」的患者层面 bootstrap CI（H1 判据）。
5. T2 各对比病种（DKD vs 单一病种）AUROC 与 Δ，Bonferroni 校正。
6. H4：T1 中 RNA-seq 测试队列的非劣效（ΔAUROC 下界 > −margin）。

产出：results/09_compare/{per_fold_metrics,paired_delta,pooled_auroc,pooled_delta,t2_per_comparator,
      h4_noninferiority}.tsv、PROVENANCE.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # 共享服务器：限制 BLAS 线程

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import stats as S  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "09_compare"
KEYS = ["task", "compartment", "fold"]
PRED_FILES = [ROOT / "results" / "06_baselines" / "predictions.tsv",
              ROOT / "results" / "08_models" / "predictions.tsv"]
THR_FILES = [p.with_name("thresholds.tsv") for p in PRED_FILES]


def load_ensembled() -> tuple[pd.DataFrame, pd.DataFrame]:
    pr = pd.concat([pd.read_csv(p, sep="\t", dtype={"fold": str}) for p in PRED_FILES if p.exists()])
    th = pd.concat([pd.read_csv(p, sep="\t", dtype={"fold": str}) for p in THR_FILES if p.exists()])
    ens = pr.groupby(KEYS + ["model", "sample_uid"], as_index=False).agg(
        y_true=("y_true", "first"), score=("score", "mean"), n_seeds=("seed", "nunique"))
    thr = th.groupby(KEYS + ["model"], as_index=False)["threshold"].mean()
    return ens, thr


def expand_pairs(models: set[str]) -> list[tuple[str, str]]:
    pairs = []
    fracs = [f for f in CFG["eval"]["label_fracs"] if f < 1.0]
    for a, b in CFG["compare"]["pairs"]:
        for suf in [""] + [f"@lf{f:g}" for f in fracs]:
            if a + suf in models and b + suf in models:
                pairs.append((a + suf, b + suf))
    return pairs


def fold_tables(ens, thr, samples, tech_re) -> tuple[list, list, dict]:
    """每个 fold：点估计 + bootstrap（所有模型共享索引）。返回 per-fold 行、配对行、bootstrap 缓存。"""
    per, paired, cache = [], [], {}
    for (task, comp, fold), g in ens.groupby(KEYS, sort=True):
        wide = g.pivot(index="sample_uid", columns="model", values="score").dropna(axis=0, how="any")
        ys = g.drop_duplicates("sample_uid").set_index("sample_uid").loc[wide.index, "y_true"].to_numpy()
        models = list(wide.columns)
        clusters = samples.loc[wide.index, "dup_group"].to_numpy()
        n1, n0 = int(ys.sum()), int(len(ys) - ys.sum())
        pt = dict(zip(models, S.auc_rows(ys, wide.to_numpy().T)))
        boot = S.boot_auc(ys, wide.to_numpy().T, clusters, CFG["eval"]["n_boot"], CFG["seed"])
        techs = samples.loc[wide.index, "technology"].astype(str).unique()
        tech = ";".join(sorted(techs))
        cache[(task, comp, fold)] = {"boot": boot, "models": models, "y": ys, "wide": wide,
                                     "rnaseq": bool(re.search(tech_re, tech))}
        tsub = thr[(thr.task == task) & (thr.compartment == comp) & (thr.fold == fold)].set_index("model")
        for j, m in enumerate(models):
            pm = S.point_metrics(ys, wide[m].to_numpy(),
                                 float(tsub.threshold[m]) if m in tsub.index else None)
            lo, hi = S.ci(boot[:, j])
            la, lv = S.logit_auc_var(boot[:, j], pm["auroc"], n1, n0)
            per.append({"task": task, "compartment": comp, "fold": fold, "technology": tech, "model": m,
                        **pm, "auroc_lo": lo, "auroc_hi": hi, "logit_auroc_cc": la, "var_logit_auroc": lv,
                        "n_seeds": int(g[g.model == m].n_seeds.max())})
        for a, b in expand_pairs(set(models)):
            ia, ib = models.index(a), models.index(b)
            d = boot[:, ia] - boot[:, ib]
            lo, hi = S.ci(d)
            paired.append({"task": task, "compartment": comp, "fold": fold, "technology": tech,
                           "model_a": a, "model_b": b,
                           "delta": float(pt[a] - pt[b]),
                           "lo": lo, "hi": hi, "p_boot": S.boot_p_two_sided(d),
                           "var_delta": S.delta_var(d, pt[a], pt[b], n1, n0), "n_boot_valid": int((~np.isnan(d)).sum())})
    return per, paired, cache


def pooled(per: pd.DataFrame, paired: pd.DataFrame, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = [("task_compartment", ["task", "compartment"]), ("task", ["task"])]
    pa, pdl = [], []
    for lvl, by in levels:
        for key, g in per.groupby(by + ["model"]):
            r = S.dersimonian_laird(g.logit_auroc_cc.to_numpy(), g.var_logit_auroc.to_numpy())
            row = dict(zip(by + ["model"], key if isinstance(key, tuple) else (key,)))
            row.update({"level": lvl, "k": r.get("k", 0)})
            for f in ("mu", "lo", "hi", "pi_lo", "pi_hi"):
                row[f"auroc_{f}"] = float(S.expit(r[f])) if f in r and not np.isnan(r[f]) else np.nan
            row.update({"tau2_logit": r.get("tau2", np.nan), "I2": r.get("I2", np.nan)})
            pa.append(row)
        for key, g in paired.groupby(by + ["model_a", "model_b"]):
            r = S.dersimonian_laird(g.delta.to_numpy(), g.var_delta.to_numpy())
            # 宏平均 Δ 的 bootstrap：每个 fold 独立重抽样，第 b 次重复取各 fold Δ_b 的均值
            ds = []
            for _, fr in g.iterrows():
                c = cache[(fr.task, fr.compartment, fr.fold)]
                ds.append(c["boot"][:, c["models"].index(fr.model_a)] - c["boot"][:, c["models"].index(fr.model_b)])
            macro = np.nanmean(np.stack(ds), axis=0) if ds else np.array([np.nan])
            lo, hi = S.ci(macro)
            row = dict(zip(by + ["model_a", "model_b"], key))
            row.update({"level": lvl, "k": r.get("k", 0), "dl_delta": r.get("mu", np.nan),
                        "dl_lo": r.get("lo", np.nan), "dl_hi": r.get("hi", np.nan),
                        "dl_p": r.get("p", np.nan), "pi_lo": r.get("pi_lo", np.nan),
                        "pi_hi": r.get("pi_hi", np.nan), "tau2": r.get("tau2", np.nan),
                        "I2": r.get("I2", np.nan), "macro_delta": float(g.delta.mean()),
                        "macro_lo": lo, "macro_hi": hi, "macro_p": S.boot_p_two_sided(macro),
                        "criterion_met": bool(r.get("mu", np.nan) > 0 and lo > 0)})
            pdl.append(row)
    return pd.DataFrame(pa), pd.DataFrame(pdl)


def t2_comparators(ens: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """DKD vs 每个对比病种：各 fold AUROC → DL 合并；Δ（相对 B1）fold 内配对 bootstrap → DL；Bonferroni。"""
    if "T2" not in CFG["tasks"]:
        return pd.DataFrame()
    diag = D.load_diagnosis()
    g2 = ens[ens.task == "T2"].copy()
    if g2.empty:
        return pd.DataFrame()
    g2["diagnosis"] = diag.reindex(g2.sample_uid).to_numpy()
    comps = [c for c in CFG["tasks"]["T2"]["neg"] if (g2.diagnosis == c).any()]
    ref = "B1-rankLASSO"
    rows = []
    for dz in comps:
        per_fold = []
        for (task, comp, fold), g in g2.groupby(KEYS):
            sub = g[g.diagnosis.isin(CFG["tasks"]["T2"]["pos"] + [dz])]
            wide = sub.pivot(index="sample_uid", columns="model", values="score").dropna()
            y = sub.drop_duplicates("sample_uid").set_index("sample_uid").loc[wide.index, "y_true"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            boot = S.boot_auc(y, wide.to_numpy().T, samples.loc[wide.index, "dup_group"].to_numpy(),
                              CFG["eval"]["n_boot"], CFG["seed"])
            pt = S.auc_rows(y, wide.to_numpy().T)
            n1, n0 = int(y.sum()), int(len(y) - y.sum())
            cols = list(wide.columns)
            for j, m in enumerate(cols):
                rname = ref + (m[m.index("@"):] if "@" in m else "")   # 同一 label fraction 的 B1
                jr = cols.index(rname) if rname in cols and rname != m else None
                la, lv = S.logit_auc_var(boot[:, j], float(pt[j]), n1, n0)
                d = boot[:, j] - boot[:, jr] if jr is not None else None
                per_fold.append({"model": m, "fold": fold, "compartment": comp, "n_comp": n0,
                                 "auroc": float(pt[j]), "logit_cc": la, "var_logit": lv,
                                 "delta": float(pt[j] - pt[jr]) if d is not None else np.nan,
                                 "var_delta": S.delta_var(d, float(pt[j]), float(pt[jr]), n1, n0)
                                 if d is not None else np.nan})
        pf = pd.DataFrame(per_fold)
        for m, g in (pf.groupby("model") if len(pf) else []):
            ra = S.dersimonian_laird(g.logit_cc.to_numpy(), g.var_logit.to_numpy())
            rd = S.dersimonian_laird(g.delta.to_numpy(), g.var_delta.to_numpy()) if g.delta.notna().any() else {}
            rows.append({"comparator": dz, "model": m, "k": ra.get("k", 0), "n_comp_total": int(g.n_comp.sum()),
                         "auroc": float(S.expit(ra["mu"])) if "mu" in ra else np.nan,
                         "auroc_lo": float(S.expit(ra["lo"])) if "lo" in ra else np.nan,
                         "auroc_hi": float(S.expit(ra["hi"])) if "hi" in ra else np.nan,
                         "delta_vs_B1": rd.get("mu", np.nan), "delta_lo": rd.get("lo", np.nan),
                         "delta_hi": rd.get("hi", np.nan), "p": rd.get("p", np.nan)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonferroni"] = np.minimum(1.0, out.p * len(comps))
    return out


def h4(paired: pd.DataFrame) -> pd.DataFrame:
    m = CFG["eval"]["noninferiority_margin"]
    sub = paired[(paired.task == "T1") & paired.technology.str.contains(CFG["compare"]["rnaseq_technology_regex"],
                                                                         regex=True)]
    sub = sub[(sub.model_a == "RRG-full") & (sub.model_b == "B1-rankLASSO")].copy()
    sub["margin"] = -m
    sub["noninferior"] = sub.lo > -m
    return sub


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    samples = D.load_samples_unlabelled()
    ens, thr = load_ensembled()
    per, paired, cache = fold_tables(ens, thr, samples, CFG["compare"]["rnaseq_technology_regex"])
    per, paired = pd.DataFrame(per), pd.DataFrame(paired)
    pa, pdl = pooled(per, paired, cache)
    t2 = t2_comparators(ens, samples)
    nh = h4(paired) if len(paired) else pd.DataFrame()
    outs = {"per_fold_metrics": per, "paired_delta": paired, "pooled_auroc": pa, "pooled_delta": pdl,
            "t2_per_comparator": t2, "h4_noninferiority": nh}
    paths = []
    for name, df in outs.items():
        p = OUT / f"{name}.tsv"
        df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    write_provenance("09_compare", [p for p in PRED_FILES + THR_FILES if p.exists()]
                     + [ROOT / "data" / "processed" / "samples.tsv"], paths, CFG["seed"],
                     {"n_boot": CFG["eval"]["n_boot"], "n_folds": len(cache)})
    print(f"[09] {len(per)} fold×model 行，{len(paired)} 配对行 → {OUT}")


if __name__ == "__main__":
    main()
