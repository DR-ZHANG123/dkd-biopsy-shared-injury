"""Stage 21 A5：跨队列相似度（特异性比）的患者 bootstrap 95% CI、min_r_dd 阈值敏感性、阈值写入时间的 git 核查。

特异性比 = r(DKD_A, 其他CKD_B) / r(DKD_A, DKD_B)（Spearman，与 stage 11b 相同的 shift 与基因交集）。
bootstrap：每次在每个单元内按组（DKD / 其他 CKD / 对照）有放回重抽样病人（同一次迭代所有有序对共用该单元的重抽样，
因此跨对的依赖在「中位数 / 比值和」的 CI 中得到保留）。汇总量：信息对的中位数；不设阈值的比值和 Σr_do / Σr_dd。
git：config/run.yaml 中 min_r_dd 首次出现的 commit 与 results/11b_shared_axis/specificity_ratio.tsv 首次出现的 commit，
以及更早的 pilot 记录（notes/10_research_idea.md）中已报告的特异性比。
产出 results/21_robustness/A5_ratio/{ratio_ci.tsv, ratio_summary.tsv, threshold_history.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import NON_DISEASE, full_ranks  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, units  # noqa: E402

O = OUT / "A5_ratio"
SA = CFG["shared_axis_tests"]


def unit_data(u, s, U):
    R = full_ranks(U[u], s).dropna(axis=1)
    d = s.loc[R.index, "diagnosis"].to_numpy()
    grp = {"CTRL": np.flatnonzero(d == "CONTROL"), "DKD": np.flatnonzero(d == "DKD"),
           "OTH": np.flatnonzero(~np.isin(d, list(NON_DISEASE)))}
    return R, grp


def shifts(R: pd.DataFrame, grp: dict, pick: dict | None) -> dict[str, pd.Series]:
    X = R.to_numpy()
    idx = pick or grp
    c = X[idx["CTRL"]].mean(0)
    return {k: pd.Series(X[idx[k]].mean(0) - c, index=R.columns) for k in ("DKD", "OTH")}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pair_stats(sa: dict, sb: dict) -> tuple[float, float]:
    g = sa["DKD"].index.intersection(sb["DKD"].index)
    da = sa["DKD"][g].to_numpy()
    return spearman(da, sb["DKD"][g].to_numpy()), spearman(da, sb["OTH"][g].to_numpy())


def summarise(rdd: np.ndarray, rdo: np.ndarray, thr: float) -> dict:
    inf = rdd >= thr
    ratio = rdo / rdd
    return {"n_informative": int(inf.sum()), "median_ratio": float(np.median(ratio[inf])) if inf.any() else np.nan,
            "min_ratio": float(ratio[inf].min()) if inf.any() else np.nan,
            "max_ratio": float(ratio[inf].max()) if inf.any() else np.nan,
            "ratio_of_sums": float(rdo[inf].sum() / rdd[inf].sum()) if inf.any() else np.nan}


def git_history() -> pd.DataFrame:
    def first(args):
        o = subprocess.run(["git", "-C", str(ROOT), "log", "--reverse", "--format=%h|%ad|%s", "--date=iso"] + args,
                           capture_output=True, text=True).stdout.strip().splitlines()
        return o[0].split("|", 2) if o else ["", "", ""]
    rows = []
    for what, args in (("config_min_r_dd_first_commit", ["-S", "min_r_dd", "--", "config/run.yaml"]),
                       ("specificity_ratio_tsv_first_commit", ["--", "results/11b_shared_axis/specificity_ratio.tsv"]),
                       ("stage11b_script_first_commit", ["--", "scripts/stages/11b_shared_axis_tests.py"]),
                       ("pilot_ratio_in_notes_first_commit", ["-S", "特异性比", "--", "notes/10_research_idea.md"])):
        h, d, m = first(args)
        rows.append({"item": what, "commit": h, "date": d, "message": m})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    pairs = [tuple(p) for p in SA["ratio_pairs"]]
    names = sorted({x for p in pairs for x in p})
    data = {u: unit_data(u, s, U) for u in names}
    base = {u: shifts(*data[u], None) for u in names}
    obs = np.array([pair_stats(base[a], base[b]) for a, b in pairs])
    B = []
    for _ in range(RV["n_boot"]):
        sh = {}
        for u in names:
            R, grp = data[u]
            sh[u] = shifts(R, grp, {k: rng.choice(v, len(v)) for k, v in grp.items()})
        B.append([pair_stats(sh[a], sh[b]) for a, b in pairs])
    B = np.array(B)                                              # boot × pair × (r_dd, r_do)
    rows = []
    for j, (a, b) in enumerate(pairs):
        rb = B[:, j, 1] / B[:, j, 0]
        rows.append({"unit_A_dkd": a, "unit_B": b, "r_dkdA_dkdB": obs[j, 0], "r_dkdA_otherB": obs[j, 1],
                     "ratio": obs[j, 1] / obs[j, 0], "ratio_lo": np.percentile(rb, 2.5), "ratio_hi": np.percentile(rb, 97.5),
                     "r_dd_lo": np.percentile(B[:, j, 0], 2.5), "r_dd_hi": np.percentile(B[:, j, 0], 97.5),
                     "informative_at_0.2": bool(obs[j, 0] >= SA["min_r_dd"])})
    C = pd.DataFrame(rows)
    ref = pd.read_csv(ROOT / "results/11b_shared_axis/specificity_ratio.tsv", sep="\t")
    assert np.allclose(C.ratio.to_numpy(), ref.specificity_ratio.to_numpy(), atol=1e-4), "与 stage 11b 不一致"
    summ = []
    for thr in RV["ratio_thresholds"]:
        o = summarise(obs[:, 0], obs[:, 1], thr)
        bs = [summarise(B[i, :, 0], B[i, :, 1], thr) for i in range(len(B))]
        for k in ("median_ratio", "ratio_of_sums"):
            v = np.array([x[k] for x in bs], float)
            o[f"{k}_lo"], o[f"{k}_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        summ.append({"min_r_dd": thr} | o)
    S = pd.DataFrame(summ)
    H = git_history()
    outs = {"ratio_ci.tsv": C, "ratio_summary.tsv": S, "threshold_history.tsv": H}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("21_robustness/A5_ratio", [ROOT / "results/11b_shared_axis/specificity_ratio.tsv"], [O / f for f in outs],
                     CFG["seed"], {"n_boot": RV["n_boot"], "thresholds": RV["ratio_thresholds"]})
    with pd.option_context("display.width", 250):
        print(C.round(3).to_string())
        print(S.round(3).to_string())
        print(H.to_string())


if __name__ == "__main__":
    main()
