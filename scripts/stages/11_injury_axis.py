"""Stage 11：留一队列的通用损伤轴、样本损伤分数、病种偏移在损伤轴上的占比。

对每个评估单元 u：损伤轴只由同区室其他来源单元估计（非 DKD 病种 vs 对照；不用 DKD 标签）。
产出 results/11_injury/
  axis_<comp>_<unit>.tsv          逐基因损伤效应量
  injury_scores.tsv               sample_uid, eval_unit, compartment, diagnosis, injury
  shift_decomposition.tsv         各评估单元内每个病种相对对照的偏移落在外部损伤轴上的比例
  axis_concordance.tsv            各来源单元效应量两两 Spearman（损伤轴是否跨队列/平台/技术一致）
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import NON_DISEASE, effect_sizes, full_ranks, injury_axis, injury_score  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "11_injury"


def load_samples() -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    s = pd.read_csv(ROOT / "data/processed/samples.tsv", sep="\t").set_index("sample_uid")
    c = pd.read_csv(ROOT / "data/processed/cohorts.tsv", sep="\t").set_index("sample_uid")
    units = {k: g.index for k, g in c.groupby("cohort")}
    rest = s[~s.index.isin(c.index)]
    for ser, g in rest.groupby("series"):
        keep = g[g.keep_for_pretrain | g.role.str.startswith(("eval", "external"))]
        # 同一 dup_group 在单元内只留一个
        units.setdefault(ser, keep[~keep.dup_group.duplicated()].index)
    return s, units


def shift_share(R: pd.DataFrame, diag: pd.Series, axis: pd.Series) -> list[dict]:
    ax = axis.reindex(R.columns).dropna()
    R = R[ax.index]
    u = (ax / np.linalg.norm(ax)).values
    ctrl = (diag == "CONTROL").values
    rows = []
    if ctrl.sum() < 3:
        return rows
    for k in sorted(set(diag) - (NON_DISEASE - {"DKD"})):
        m = (diag == k).values
        if m.sum() < 3:
            continue
        sh = R.values[m].mean(0) - R.values[ctrl].mean(0)
        rows.append({"diagnosis": k, "n": int(m.sum()), "n_ctrl": int(ctrl.sum()),
                     "share_on_injury_axis": float((sh @ u) ** 2 / (sh @ sh)),
                     "cosine": float(sh @ u / np.linalg.norm(sh))})
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    s, units = load_samples()
    scores, decomp, conc, outs = [], [], [], []
    for comp, spec in CFG["injury"]["compartments"].items():
        # 来源单元之间的一致性（全部来源、各自单独估计）
        effs = {}
        for u in spec["source"]:
            R = full_ranks(units[u], s).dropna(axis=1)
            g = effect_sizes(R, s.loc[units[u], "diagnosis"])
            if g is not None:
                effs[u] = g
        for a, b in itertools.combinations(effs, 2):
            gi = effs[a].index.intersection(effs[b].index)
            conc.append({"compartment": comp, "unit_a": a, "unit_b": b, "n_genes": len(gi),
                         "spearman": spearmanr(effs[a][gi], effs[b][gi])[0]})
        for ev in spec["eval"]:
            src = [units[u] for u in spec["source"] if u != ev]
            axis = injury_axis(src, s)
            p = OUT / f"axis_{comp}_{ev}.tsv"
            axis.to_csv(p, sep="\t")
            outs.append(p)
            idx = units[ev]
            R = full_ranks(idx, s).dropna(axis=1)
            inj = injury_score(R, axis)
            scores.append(pd.DataFrame({"sample_uid": idx, "eval_unit": ev, "compartment": comp,
                                        "diagnosis": s.loc[idx, "diagnosis"].values, "injury": inj.values}))
            for r in shift_share(R, s.loc[idx, "diagnosis"], axis):
                decomp.append({"compartment": comp, "eval_unit": ev, **r})
    pd.concat(scores).to_csv(OUT / "injury_scores.tsv", sep="\t", index=False)
    pd.DataFrame(decomp).to_csv(OUT / "shift_decomposition.tsv", sep="\t", index=False)
    pd.DataFrame(conc).to_csv(OUT / "axis_concordance.tsv", sep="\t", index=False)
    outs += [OUT / f for f in ("injury_scores.tsv", "shift_decomposition.tsv", "axis_concordance.tsv")]
    write_provenance("11_injury", [ROOT / "data/processed/samples.tsv"], outs, CFG["seed"])
    print(pd.DataFrame(conc).round(2).to_string())
    print(pd.DataFrame(decomp).round(2).to_string())


if __name__ == "__main__":
    main()
