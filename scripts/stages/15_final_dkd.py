"""Stage 15 final：冻结设计（results/15_model/FROZEN_DESIGN.json）在 DKD vs 其他 CKD 上的一次性评估。

协议与 dev 的 E1 相同：测试单元 = ERCB 各批次（肾小球 H1/H7、肾小管 H1/H7），训练 = 其余全部单元
（去掉与测试单元共享标本的样本）；测试集 = 测试单元内的全部病人（不含对照），阳性 = DKD。
主指标：共享轴调整 AUROC；次指标：原始 AUROC、stratum 内 AUROC。只运行一次，已有预测则拒绝重跑。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.m2_baselines import run_baseline  # noqa: E402
from lib.m2_data import M2, features, make_folds, resolve_method  # noqa: E402
from lib.m2_eval import cell_metrics, eval_mask  # noqa: E402
from lib.m2_train import fit_predict, heads_for  # noqa: E402
from lib.repro import ROOT, config_sha, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "15_model" / "final_dkd"
TEST_UNITS = {"GLOM": ["ERCB_GLOM_H1", "ERCB_GLOM_H7"], "TUB": ["ERCB_TUB_H1", "ERCB_TUB_H7"]}


def dev15():
    from importlib import import_module
    return import_module("15_dev")


def rows_for(fold, score: np.ndarray, a: pd.Series, method: str) -> pd.DataFrame:
    m = eval_mask_dkd(fold.test)
    te = fold.test[m]
    return pd.DataFrame({"compartment": fold.comp, "test_unit": fold.test_unit, "method": method,
                         "sample_uid": te.index, "diagnosis": te.diagnosis.values, "stratum": te.stratum.values,
                         "y": (te.diagnosis == "DKD").astype(int).values, "score": np.asarray(score)[m],
                         "a": a.reindex(te.index).values})


def eval_mask_dkd(test: pd.DataFrame) -> np.ndarray:
    """与 dev 的 eval_mask 相同（测试单元内病人，不含对照），但保留 DKD 作为阳性。"""
    return test.role.isin(["head", "neg"]).to_numpy()


def main() -> None:
    from lib.freeze import check_frozen
    frozen = check_frozen()
    if (OUT / "predictions.tsv").exists():
        raise SystemExit("DKD 最终评估已运行过，按协议不重跑")
    set_global_seed(CFG["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d15 = dev15()
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    F = features()
    preds = []
    for comp, units in TEST_UNITS.items():
        for fold in make_folds("E1", comp, drop_test_dkd=False, test_units=units):
            n_pos = int((fold.test.diagnosis == "DKD").sum())
            print(f"{fold.key}: test DKD={n_pos} test patients={int(eval_mask_dkd(fold.test).sum())} "
                  f"train DKD={int((fold.train.diagnosis == 'DKD').sum())}", flush=True)
            # 冻结 RRG-ID
            mc = resolve_method(frozen["primary_method"])
            Xtr, Xte, a, a_std = d15.fold_arrays(fold, F, mc)
            heads = heads_for(fold.train, M2["labels"]["head"], M2["eval"]["min_train_pos"])
            assert "DKD" in heads, heads
            sc, info = fit_predict(Xtr, Xte, fold.train, a.loc[fold.train.index].to_numpy(np.float32),
                                   a_std.loc[fold.train.index].to_numpy(np.float32), d15.programs_for(mc),
                                   heads, mc, CFG["seed"] % 100000, dev, False)
            preds.append(rows_for(fold, sc["DKD"].to_numpy(), a, frozen["primary_method"]))
            # 对照
            Xtr, Xte, a, _ = d15.fold_arrays(fold, F, None)
            M = d15.programs_for(M2["defaults"])
            for b in frozen["comparators"]:
                s, _ = run_baseline(b, Xtr, Xte, fold.train, "DKD", a.loc[fold.train.index].to_numpy(),
                                    a.loc[fold.test.index].to_numpy(), M, CFG["seed"])
                preds.append(rows_for(fold, s, a, b))
    P = pd.concat(preds)
    P.to_csv(OUT / "predictions.tsv", sep="\t", index=False)
    met = []
    for (comp, u, m), g in P.groupby(["compartment", "test_unit", "method"]):
        r = cell_metrics(g.y.to_numpy(), g.score.to_numpy(), g.a.to_numpy(), g.stratum.to_numpy())
        met.append({"compartment": comp, "test_unit": u, "method": m, "n_dkd": int(g.y.sum()),
                    "n_other": int((1 - g.y).sum()), **r})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "metrics.tsv", sep="\t", index=False)
    write_provenance("15_model/final_dkd", [ROOT / "results/15_model/FROZEN_DESIGN.json"],
                     [OUT / "predictions.tsv", OUT / "metrics.tsv"], CFG["seed"], {"frozen": frozen["frozen_at"]})
    print(met.round(3).to_string())


if __name__ == "__main__":
    main()
