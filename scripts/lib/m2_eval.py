"""Stage 15 评估：评估单元格（fold × 病种）、AUROC / 共享轴调整 AUROC / stratum 内 AUROC、DKD 隔离断言。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.injury import adjusted_auroc
from lib.repro import load_config

CFG = load_config()
EV = CFG["model2"]["eval"]


def assert_no_dkd_eval(test: pd.DataFrame, disease: str | None = None) -> None:
    """dev 阶段：DKD 既不能是阳性类，也不能出现在任何评估集中。"""
    if disease == "DKD":
        raise AssertionError("dev 阶段禁止评估 DKD")
    if (test.diagnosis == "DKD").any():
        raise AssertionError(f"dev 评估集中出现 {int((test.diagnosis == 'DKD').sum())} 个 DKD 样本")


def eval_mask(test: pd.DataFrame) -> np.ndarray:
    """评估集 = 测试单元内的病人（head 类 + neg_only），不含对照、DKD。"""
    return (test.role.isin(["head", "neg"]) & (test.diagnosis != "DKD")).to_numpy()


def cells_for(test: pd.DataFrame, train: pd.DataFrame, diseases: list[str] | None = None) -> list[str]:
    m = eval_mask(test)
    vc_te = test[m].diagnosis.value_counts()
    vc_tr = train.diagnosis.value_counts()
    out = []
    for d in CFG["model2"]["labels"]["head"]:
        if d == "DKD" or (diseases and d not in diseases):
            continue
        npos = vc_te.get(d, 0)
        if npos >= EV["min_test_pos"] and m.sum() - npos >= EV["min_test_neg"] \
                and vc_tr.get(d, 0) >= EV["min_train_pos"]:
            out.append(d)
    return out


def within_strata_auroc(y: np.ndarray, score: np.ndarray, strata: np.ndarray) -> float:
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        yy = y[m]
        w = yy.sum() * (len(yy) - yy.sum())
        if w:
            num += w * roc_auc_score(yy, score[m])
            den += w
    return float(num / den) if den else float("nan")


def cell_metrics(y: np.ndarray, score: np.ndarray, a: np.ndarray, strata: np.ndarray) -> dict:
    ok = ~np.isnan(score)
    y, score, a, strata = y[ok], score[ok], a[ok], strata[ok]
    if len(np.unique(y)) < 2:
        return {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auroc": np.nan, "adj_auroc": np.nan,
                "auroc_ws": np.nan}
    return {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auroc": float(roc_auc_score(y, score)),
            "adj_auroc": adjusted_auroc(y, score, a), "auroc_ws": within_strata_auroc(y, score, strata)}


def metrics_from_preds(P: pd.DataFrame) -> pd.DataFrame:
    """P 列：family compartment test_unit method sample_uid diagnosis stratum disease score a（E3 为拼接后的 OOF）。"""
    rows = []
    for (fam, comp, unit, meth, d), g in P.groupby(["family", "compartment", "test_unit", "method", "disease"]):
        if d == "DKD" or (g.diagnosis == "DKD").any():
            raise AssertionError("dev 预测中出现 DKD")
        y = (g.diagnosis.to_numpy() == d).astype(int)
        rows.append({"family": fam, "compartment": comp, "test_unit": unit, "method": meth, "disease": d,
                     **cell_metrics(y, g.score.to_numpy(float), g.a.to_numpy(float), g.stratum.to_numpy())})
    return pd.DataFrame(rows)
