"""Stage 17：冻结 RRG-ID 在独立测试队列上的一次性评估（严格按 results/17_independent/PLAN.json，commit 1ab2135）。

- 冻结校验：lib.freeze.check_frozen()（冻结提交时的 config 各段未改动、model2.defaults = FROZEN_DESIGN）；
  PLAN.json 必须与预注册提交一致；已有 predictions.tsv 则拒绝重跑。
- 训练：stage 15 E1 协议下该区室全部 model2 单元（unit_table），去掉与任何测试队列表达判重的样本
  （stage 17a overlap_pairs 的 expr_dup，整 dup_group 去掉）；超参全部取冻结设计，不改动。
  每个区室模型只训练一次，同一组分数用于该区室全部测试队列的全部对比（每个队列每个对比只打分一次）。
- 测试：GSE162830 → GLOM 模型；KPMP 活检切片 bulk、GSE166239 → TUB 模型（WHOLE 切片，同 GSE142025 规则）。
  特征 = 队列内样本内百分位秩（与 stage 01/04 同法），未测到的基因用训练集中位数填补。
- 共享轴：来源 = config injury.compartments[comp].source 全部训练单元（测试队列不作来源），测试样本在其自身秩上打分。
- 指标：原始 AUROC、共享轴调整 AUROC（Janes–Pepe，lib.injury）；患者层面配对 bootstrap（n = eval.n_boot）Δ vs B-L2、B-cPCA。
用法：python scripts/stages/17_independent_test.py [cuda:0]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.evalkit import median_impute_apply, median_impute_fit  # noqa: E402
from lib.freeze import check_frozen  # noqa: E402
from lib.indep_eval import (COHORTS, cohort_ranks, contrast_metrics, select_test,  # noqa: E402
                            test_axis_score)
from lib.m2_baselines import run_baseline  # noqa: E402
from lib.m2_data import M2, _axis, _unit_score, features, resolve_method, standardize_within, unit_table, universe  # noqa: E402
from lib.m2_train import fit_predict, heads_for  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "17_independent"
CAND = OUT / "candidates"
PLAN_COMMIT = "1ab2135"
REFS = ["B-L2", "B-cPCA"]


def check_plan() -> dict:
    cur = (OUT / "PLAN.json").read_text()
    reg = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PLAN_COMMIT}:results/17_independent/PLAN.json"]).decode()
    if json.loads(cur) != json.loads(reg):
        raise SystemExit("PLAN.json 与预注册提交不一致，拒绝运行")
    return json.loads(cur)


def training_table(comp: str, test_uids: set[str]) -> tuple[pd.DataFrame, int]:
    t = unit_table(comp)
    P = pd.read_csv(CAND / "overlap_pairs.tsv", sep="\t")
    P = P[P.expr_dup & (P.series_a != P.series_b)]
    hit = set()
    for _, r in P.iterrows():
        if r.gsm_a in test_uids and r.series_b not in COHORTS:
            hit.add(f"{r.series_b}|{r.gsm_b}")
    groups = set(t.loc[t.index.intersection(list(hit)), "dup_group"])
    drop = t.dup_group.isin(groups)
    return t[~drop].copy(), int(drop.sum())


def main() -> None:
    frozen = check_frozen()
    check_plan()
    if (OUT / "predictions.tsv").exists():
        raise SystemExit("独立测试已运行过，按协议不重跑")
    set_global_seed(CFG["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    S = pd.read_csv(CAND / "candidate_samples.tsv", sep="\t", index_col=0, low_memory=False)
    tests = {c: select_test(S, c) for c in COHORTS}
    genes = universe()
    F = features()
    d15 = import_module("15_dev")
    preds, train_info = [], []
    for comp in ("GLOM", "TUB"):
        cohorts = [c for c, r in COHORTS.items() if r["model"] == comp]
        T = pd.concat([tests[c] for c in cohorts])
        train, n_drop = training_table(comp, set(T.index))
        Ftr = F.loc[train.index].to_numpy(np.float32)
        med = median_impute_fit(Ftr)
        Xtr = median_impute_apply(Ftr, med)
        src = tuple(u for u in CFG["injury"]["compartments"][comp]["source"])
        assert not set(src) & set(COHORTS)
        axis = _axis(comp, src)
        a_tr = pd.concat([_unit_score(comp, src, u) for u in sorted(set(train.unit))])
        a_tr = a_tr[~a_tr.index.duplicated()].reindex(train.index)
        assert a_tr.notna().all()
        Xte, a_te = [], []
        for c in cohorts:
            ex = pd.read_parquet(ROOT / CFG["paths"]["interim"] / "indep" / f"{c}_expr.parquet")[tests[c].index]
            R = cohort_ranks(ex)
            print(f"{comp} {c}: n={len(R)} genes ranked={R.shape[1]} universe coverage="
                  f"{R.columns.isin(genes).sum()}/{len(genes)}", flush=True)
            Xte.append(median_impute_apply(R.reindex(columns=genes).to_numpy(np.float32), med))
            a_te.append(test_axis_score(R, axis))
        Xte, a_te = np.vstack(Xte), pd.concat(a_te).reindex(T.index)
        mc = resolve_method(frozen["primary_method"])
        grp = train["stratum" if mc["compare_level"] == "stratum" else "unit"]
        a_std = standardize_within(a_tr, grp)
        heads = heads_for(train, M2["labels"]["head"], M2["eval"]["min_train_pos"])
        assert "DKD" in heads, heads
        train_info.append({"model": comp, "n_train": len(train), "n_train_dropped_dup": n_drop,
                           "n_train_dkd": int((train.diagnosis == "DKD").sum()), "heads": ",".join(heads),
                           "axis_sources": ",".join(src), "test_cohorts": ",".join(cohorts)})
        print(train_info[-1], flush=True)
        scores = {}
        sc, _ = fit_predict(Xtr, Xte, train, a_tr.to_numpy(np.float32), a_std.to_numpy(np.float32),
                            d15.programs_for(mc), heads, mc, CFG["seed"] % 100000, dev, False)
        scores[frozen["primary_method"]] = sc["DKD"].to_numpy()
        M = d15.programs_for(M2["defaults"])
        for b in frozen["comparators"]:
            scores[b], _ = run_baseline(b, Xtr, Xte, train, "DKD", a_tr.to_numpy(), a_te.to_numpy(), M, CFG["seed"])
        for m, s in scores.items():
            preds.append(pd.DataFrame({"model": comp, "cohort": T.cohort.values, "method": m, "sample_uid": T.index,
                                       "participant": T.participant.values, "diagnosis": T.diagnosis.values,
                                       "score": np.asarray(s, float), "a": a_te.values}))
    P = pd.concat(preds)
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT / "predictions.tsv", sep="\t", index=False)
    pd.DataFrame(train_info).to_csv(OUT / "training_summary.tsv", sep="\t", index=False)
    pd.concat(tests.values()).to_csv(OUT / "test_samples.tsv", sep="\t")
    methods = [frozen["primary_method"]] + list(frozen["comparators"])
    mets, dels = [], []
    for c, r in COHORTS.items():
        for contrast, neg in (("primary", r["primary_neg"]), ("secondary", r["secondary_neg"])):
            g = P[(P.cohort == c) & P.diagnosis.isin(["DKD"] + neg)].copy()
            g["y"] = (g.diagnosis == "DKD").astype(int)
            m, d = contrast_metrics(g, methods, REFS, CFG["eval"]["n_boot"], CFG["seed"])
            lab = {"cohort": c, "model": r["model"], "contrast": contrast, "negatives": "+".join(neg)}
            mets.append(m.assign(**lab))
            dels.append(d.assign(**lab))
    met = pd.concat(mets)[["cohort", "model", "contrast", "negatives", "method", "n_dkd", "n_neg", "auroc", "auroc_lo",
                           "auroc_hi", "adj_auroc", "adj_lo", "adj_hi"]]
    dl = pd.concat(dels)[["cohort", "model", "contrast", "negatives", "method", "vs", "metric", "delta", "lo", "hi",
                          "p_boot", "n_boot_valid"]]
    met.to_csv(OUT / "metrics.tsv", sep="\t", index=False)
    dl.to_csv(OUT / "paired_bootstrap_deltas.tsv", sep="\t", index=False)
    outs = [OUT / f for f in ("predictions.tsv", "metrics.tsv", "paired_bootstrap_deltas.tsv", "training_summary.tsv",
                              "test_samples.tsv")]
    write_provenance("17_independent", [ROOT / "results/15_model/FROZEN_DESIGN.json", OUT / "PLAN.json",
                                        CAND / "candidate_samples.tsv", CAND / "overlap_pairs.tsv"]
                     + sorted((ROOT / CFG["paths"]["interim"] / "indep").glob("*_expr.parquet")), outs, CFG["seed"],
                     {"frozen": frozen["frozen_at"], "plan_commit": PLAN_COMMIT, "freeze_check": "lib.freeze.check_frozen",
                      "n_boot": CFG["eval"]["n_boot"], "training": train_info})
    pd.set_option("display.width", 250)
    print(met.round(3).to_string())
    print(dl[dl.method == frozen["primary_method"]].round(3).to_string())


if __name__ == "__main__":
    main()
