"""Stage 15 dev：非 DKD 任务上的 RRG-ID / 消融 / 对照比较（设计选择只看这里）。

评估族（全部非 DKD；DKD 样本不进入任何测试集，assert_no_dkd_eval 强制）：
  E1 留一单元（与最终 DKD 协议同构）、E2 跨联盟（NEPTUNE ↔ ERCB）、E3 单元内 k 折。
预测按 (method, fold) 缓存：results/15_model/<tag>/preds/<method>/<fold>.tsv；已存在则跳过（--force 重算）。
用法：python scripts/stages/15_dev.py --families E1,E2 --compartments GLOM,TUB --methods RRG-ID,B-L2 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.m2_baselines import run_baseline  # noqa: E402
from lib.m2_data import (M2, covariate, features, impute, make_folds, resolve_method,  # noqa: E402
                         standardize_within, universe)
from lib.m2_eval import assert_no_dkd_eval, cells_for, eval_mask, metrics_from_preds  # noqa: E402
from lib.m2_programs import program_matrix  # noqa: E402
from lib.m2_train import fit_predict, heads_for  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
BASE = ROOT / "results" / "15_model"


def programs_for(mc: dict) -> np.ndarray:
    M, _ = program_matrix(mc["graph"], tuple(universe()), mc["diffuse_hops"], mc["diffuse_beta"],
                          mc["program_size"], mc.get("rewire_seed", 0),
                          mc.get("marker_source", "kpmp"))
    return M


def fold_cells(folds, fam: str) -> dict[str, list[str]]:
    """E1/E2：每 fold 自己的 cell；E3：按整个测试单元（各折测试集并集）确定 cell，各折共用。"""
    dis = M2["eval"]["e2_diseases"] if fam == "E2" else None
    if fam == "E4":
        return {f.key: [d for d in cells_for(f.test, f.train, [f.disease])] for f in folds}
    if fam != "E3":
        return {f.key: cells_for(f.test, f.train, dis) for f in folds}
    out = {}
    for u in {f.test_unit for f in folds}:
        fs = [f for f in folds if f.test_unit == u]
        te = pd.concat([f.test for f in fs])
        tr = pd.concat([fs[0].train[fs[0].train.unit != u], te])
        c = cells_for(te, tr, dis)
        for f in fs:
            out[f.key] = c
    return out


def fold_arrays(fold, F: pd.DataFrame, mc: dict | None):
    Xtr, Xte, _ = impute(F.loc[fold.train.index].to_numpy(np.float32), F.loc[fold.test.index].to_numpy(np.float32))
    a = covariate(fold)
    level = (mc or {}).get("compare_level", M2["baseline_center_level"])
    grp = pd.concat([fold.train, fold.test])["stratum" if level == "stratum" else "unit"]
    a_std = standardize_within(a, grp)
    return Xtr, Xte, a, a_std


def long_rows(fold, cells, scores: dict[str, np.ndarray], a: pd.Series, method: str) -> pd.DataFrame:
    m = eval_mask(fold.test)
    te = fold.test[m]
    out = []
    for d in cells:
        assert_no_dkd_eval(te, d)
        out.append(pd.DataFrame({"family": fold.family, "compartment": fold.comp, "test_unit": fold.test_unit,
                                 "fold_part": fold.part, "method": method, "sample_uid": te.index,
                                 "diagnosis": te.diagnosis.values, "stratum": te.stratum.values, "disease": d,
                                 "score": np.asarray(scores[d])[m], "a": a.reindex(te.index).values}))
    return pd.concat(out) if out else pd.DataFrame()


def run_rrgid(method, fold, cells, F, dev, keep: bool) -> pd.DataFrame:
    mc = resolve_method(method)
    Xtr, Xte, a, a_std = fold_arrays(fold, F, mc)
    heads = heads_for(fold.train, M2["labels"]["head"], M2["eval"]["min_train_pos"])
    M = programs_for(mc)
    seed0 = CFG["seed"] % 100000
    sc, info = fit_predict(Xtr, Xte, fold.train, a.loc[fold.train.index].to_numpy(np.float32),
                           a_std.loc[fold.train.index].to_numpy(np.float32), M, heads, mc, seed0, dev, keep)
    if keep:
        p = ROOT / "models" / "15_model2" / fold.key / f"{method}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"heads": heads, "models": info["models"], "hist": info["hist"]}, p)
    print(f"  {method} {fold.key} heads={heads} |r(z,a)|={np.mean(info['r_z_a']):.3f} "
          f"last={info['hist'][-1]}", flush=True)
    return long_rows(fold, cells, {d: sc[d].to_numpy() for d in cells}, a, method)


def _one_baseline(name, Xtr, Xte, train, d, a_tr, a_te, M, seed):
    s, _ = run_baseline(name, Xtr, Xte, train, d, a_tr, a_te, M, seed)
    return d, s


def run_base(method, fold, cells, F, n_jobs) -> pd.DataFrame:
    Xtr, Xte, a, _ = fold_arrays(fold, F, None)
    M = programs_for(M2["defaults"])          # 与冻结 RRG-ID 相同的固定程序
    res = Parallel(n_jobs=min(n_jobs, max(1, len(cells))))(
        delayed(_one_baseline)(method, Xtr, Xte, fold.train, d, a.loc[fold.train.index].to_numpy(),
                               a.loc[fold.test.index].to_numpy(), M, CFG["seed"]) for d in cells)
    return long_rows(fold, cells, dict(res), a, method)


_FCACHE: dict = {}


def _features_cached() -> pd.DataFrame:
    if "F" not in _FCACHE:
        _FCACHE["F"] = features()
    return _FCACHE["F"]


def _base_job(method: str, fam: str, comp: str, i: int, path: str) -> str:
    """对照的 fold 级并行任务：worker 内自行重建 fold 与特征（避免在进程间传大数组）。"""
    folds = make_folds(fam, comp)
    fold = folds[i]
    cells = fold_cells(folds, fam)[fold.key]
    t0 = time.time()
    rows = run_base(method, fold, cells, _features_cached(), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, sep="\t", index=False)
    return f"{fold.key} {method} cells={cells} {time.time() - t0:.1f}s"


def collect(out: Path) -> pd.DataFrame:
    files = sorted((out / "preds").glob("*/*.tsv"))
    P = pd.concat([pd.read_csv(f, sep="\t") for f in files if f.stat().st_size > 0])
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="E1,E2,E3")
    ap.add_argument("--compartments", default="GLOM,TUB")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-tag", default="dev")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=M2["n_jobs"])
    ap.add_argument("--no-collect", action="store_true")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = BASE / args.out_tag
    meths = list(M2["methods"]) + M2["baselines"] if args.methods == "all" else args.methods.split(",")
    dev = torch.device(args.device)
    F = features()
    base_jobs = []
    for fam in args.families.split(","):
        for comp in args.compartments.split(","):
            folds = make_folds(fam, comp)
            cells = fold_cells(folds, fam)
            for i, fold in enumerate(folds):
                if not cells[fold.key]:
                    continue
                assert_no_dkd_eval(fold.test)
                for m in meths:
                    p = out / "preds" / m / f"{fold.key}.tsv"
                    if p.exists() and not args.force:
                        continue
                    if m not in {**M2["methods"], **M2.get("dev_variants", {})}:
                        base_jobs.append((m, fam, comp, i, str(p)))
                        continue
                    t0 = time.time()
                    if m in {**M2["methods"], **M2.get("dev_variants", {})}:
                        keep = m == "RRG-ID" and fam == "E1" and args.out_tag == "dev"
                        rows = run_rrgid(m, fold, cells[fold.key], F, dev, keep)
                    else:
                        rows = run_base(m, fold, cells[fold.key], F, args.n_jobs)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    rows.to_csv(p, sep="\t", index=False)
                    print(f"{fam} {comp} {fold.key} {m} cells={cells[fold.key]} {time.time() - t0:.1f}s", flush=True)
    if base_jobs:
        for msg in Parallel(n_jobs=args.n_jobs, verbose=0, return_as="generator_unordered")(
                delayed(_base_job)(*j) for j in base_jobs):
            print(msg, flush=True)
    if args.no_collect:
        return
    P = collect(out)
    met = metrics_from_preds(P)
    met.to_csv(out / "dev_metrics.tsv", sep="\t", index=False)
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import_module("15_summarize").summarize(out)
    write_provenance(f"15_model/{args.out_tag}", [ROOT / "data/processed/samples.tsv",
                                                  ROOT / "results/13_celltype/celltype_markers.tsv",
                                                  ROOT / "results/05_graphs/edges_coexpr.tsv",
                                                  ROOT / "results/05_graphs/edges_ppi.tsv"],
                     [out / "dev_metrics.tsv", out / "dev_summary.tsv", out / "ablation_summary.tsv"],
                     CFG["seed"], {"methods": sorted(P.method.unique().tolist()),
                                   "families": sorted(P.family.unique().tolist()), "n_cells": int(len(met))})


if __name__ == "__main__":
    main()
