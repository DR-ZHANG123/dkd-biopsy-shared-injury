"""Stage 18a：用冻结设计（FROZEN_DESIGN.json）重训 RRG-ID，保存每个 (fold, seed) 的线性等效权重。

模型集合（每个区室分别）：
  FULL      全部带标签样本（unit_table 全体：head 病种 + neg_only + 对照）训练；共享轴来源 = 该区室全部来源单元；
  E1_<unit> 与 stage 15 E1 同构（留出该单元，去掉与之共享标本的训练样本，DKD 保留在训练集）；
            其中 ERCB 四个单元的 fold 与 stage 15 final_dkd 完全相同（同一 make_folds / fold_arrays / fit_predict /
            种子），测试分数须逐样本复现 results/15_model/final_dkd/predictions.tsv（写入 reproduction.tsv）。
每个 fold 保存 models/18_interpret/<key>.npz：beta / gp / gr（S 个种子）、训练集基因 SD、结构系数。
用法：python scripts/stages/18_interpret_train.py --comp GLOM --device cuda:0 [--keys E1_GLOM_ERCB_GLOM_H1,FULL_GLOM]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.interp import centered, model_weights, structure_corr  # noqa: E402
from lib.m2_data import M2, M2Fold, _sources, features, make_folds, unit_table  # noqa: E402
from lib.m2_train import fit_predict, heads_for  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
IC = CFG["interpret"]
MOD = ROOT / IC["models_dir"]
FINAL = ROOT / "results/15_model/final_dkd/predictions.tsv"


def frozen_design() -> dict:
    fz = json.loads((ROOT / "results/15_model/FROZEN_DESIGN.json").read_text())
    return dict(fz["design"])


def folds_for(comp: str) -> list[M2Fold]:
    t = unit_table(comp)
    full = M2Fold("FULL", comp, "NONE", -1, t.copy(), t.iloc[:0].copy(), _sources(comp, set()))
    return [full] + make_folds("E1", comp, drop_test_dkd=False)


def fold_key(f: M2Fold) -> str:
    return f"FULL_{f.comp}" if f.family == "FULL" else f.key


def run_fold(fold: M2Fold, F: pd.DataFrame, mc: dict, dev: torch.device) -> dict:
    d15 = import_module("15_dev")
    Xtr, Xte, a, a_std = d15.fold_arrays(fold, F, mc)
    heads = heads_for(fold.train, M2["labels"]["head"], M2["eval"]["min_train_pos"])
    M = d15.programs_for(mc)
    sc, info = fit_predict(Xtr, Xte, fold.train, a.loc[fold.train.index].to_numpy(np.float32),
                           a_std.loc[fold.train.index].to_numpy(np.float32), M, heads, mc,
                           CFG["seed"] % 100000, dev, True)
    W = [model_weights(m, Xtr) for m in info["models"]]
    ens = np.mean([w["score_train"] for w in W], 0)               # 训练集集成分数（与 fit_predict 同定义）
    pat = fold.train.role.isin(["head", "neg"]).to_numpy()
    grp = fold.train.stratum.to_numpy()[pat]
    Xc = centered(Xtr[pat], grp)
    Pc = centered((Xtr[pat] @ M), grp)
    Sc = centered(ens[pat], grp)
    out = {"heads": np.array(heads), "beta": np.stack([w["beta"] for w in W]), "gp": np.stack([w["gp"] for w in W]),
           "gr": np.stack([w["gr"] for w in W]), "x_sd": Xtr.std(0, ddof=1).astype(np.float32),
           "x_sd_pat_within": np.sqrt((Xc ** 2).sum(0) / max(1, pat.sum() - len(np.unique(grp)))).astype(np.float32),
           "struct_prog": structure_corr(Pc, Sc).astype(np.float32),
           "struct_gene": structure_corr(Xc, Sc).astype(np.float32),
           "r_z_a": np.array(info["r_z_a"]), "n_train": len(fold.train), "n_train_pat": int(pat.sum())}
    if len(fold.test):
        out["test_uid"] = fold.test.index.to_numpy().astype(str)
        out["test_score"] = sc.to_numpy(np.float32)
    return out


def check_reproduction(key: str, fold: M2Fold, res: dict) -> pd.DataFrame | None:
    if not (FINAL.exists() and fold.test_unit.startswith("ERCB_") and fold.family == "E1"):
        return None
    P = pd.read_csv(FINAL, sep="\t")
    P = P[(P.method == "RRG-ID") & (P.test_unit == fold.test_unit)].set_index("sample_uid")
    hi = list(res["heads"]).index("DKD")
    s = pd.Series(res["test_score"][:, hi], index=res["test_uid"]).reindex(P.index)
    return pd.DataFrame({"fold": key, "sample_uid": P.index, "score_final": P.score.values, "score_18": s.values,
                         "abs_diff": np.abs(P.score.values - s.values)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--device", default=IC["device"])
    ap.add_argument("--keys", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    mc = frozen_design()
    MOD.mkdir(parents=True, exist_ok=True)
    F = features()
    dev = torch.device(args.device)
    want = set(filter(None, args.keys.split(",")))
    for fold in folds_for(args.comp):
        key = fold_key(fold)
        if want and key not in want:
            continue
        p = MOD / f"{key}.npz"
        if p.exists() and not args.force:
            print(f"skip {key}", flush=True)
            continue
        res = run_fold(fold, F, mc, dev)
        np.savez_compressed(p, **res)
        r = check_reproduction(key, fold, res)
        if r is not None:
            r.to_csv(MOD / f"{key}.reproduction.tsv", sep="\t", index=False)
            print(f"  reproduction {key}: max|Δscore| = {r.abs_diff.max():.2e}", flush=True)
        print(f"{key}: heads={list(res['heads'])} n_train={res['n_train']} |r(z,a)|={res['r_z_a'].mean():.3f}",
              flush=True)
    if not want:
        outs = sorted(MOD.glob(f"*_{args.comp}*.npz")) + sorted(MOD.glob(f"*_{args.comp}*.reproduction.tsv"))
        rep = [pd.read_csv(p, sep="\t") for p in outs if p.suffix == ".tsv"]
        write_provenance(f"18_interpret/train_{args.comp}", [ROOT / "results/15_model/FROZEN_DESIGN.json",
                                                            ROOT / M2["features_cache"], FINAL], outs, CFG["seed"],
                         {"design": mc, "folds": [fold_key(f) for f in folds_for(args.comp)],
                          "final_dkd_reproduction_max_abs_diff": float(pd.concat(rep).abs_diff.max()) if rep else None})


if __name__ == "__main__":
    main()
