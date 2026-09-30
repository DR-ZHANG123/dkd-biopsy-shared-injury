"""Stage 20a：KPMP 小管细胞状态程序（供体内配对：状态 vs 同供体正常 PT / TAL；供体为重复单位；不读疾病标签）。

对每个数据集（snRNA 主、scRNA 复现）、每个谱系（PT、TAL）、每个非正常状态（a / fr / d / cyc）及修复失败合并（rf = a ∪ fr）：
  逐基因 平均 log2FC、d_z、配对 t、BH-FDR（全部供体）；敏感性：只用健康参考供体（REF）、只用 CKD 供体。
打分用程序 = snRNA 全部供体上 FDR < fdr 且 |lfc| ≥ lfc，按 d_z 每方向 top-n；scRNA 同号 |lfc| ≥ rep_lfc 记为复现。
产出 results/20_repair_state/programs/
  programs.tsv.gz        逐 数据集 × 谱系 × 状态 × 供体子集 × 基因 的配对统计量
  program_sets.tsv       打分用状态程序（snRNA 定义；含 scRNA 复现标记）
  program_summary.tsv    每个程序的上 / 下调基因数、跨数据集 lfc Spearman、调用重叠、REF-only 与 CKD-only 的一致性
  fractions.tsv.gz       供体 × 谱系 × 状态 的细胞数与比例（含类别、取材、eGFR 分箱）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.repair_state import (EGFR, K, OUT, RS, CFG, call_program, category_index, donor_categories,  # noqa: E402
                              paired_program, repair_label, state_fractions, state_pseudobulk)
from lib.repro import set_global_seed, write_provenance  # noqa: E402

O = OUT / "programs"


def programs_for(pb, cat) -> pd.DataFrame:
    pbs = state_pseudobulk(pb, RS["min_cells"])
    subsets = {"all": None, "REF": category_index(cat, "REF"), "CKD": category_index(cat, "CKD")}
    out = []
    for lin, d in RS["lineages"].items():
        for st in [s for s in d if s != "normal"] + [repair_label(lin)]:
            if f"{lin}|{st}" not in pbs:
                continue
            for sub, don in subsets.items():
                t = paired_program(pbs, lin, st, pb.genes, don)
                if len(t):
                    out.append(call_program(t).assign(dataset=pb.name, subset=sub))
    return pd.concat(out, ignore_index=True)


def summarise(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = P[P.subset == "all"]
    for (lin, st), g in a.groupby(["lineage", "state"]):
        sn = g[g.dataset == "snRNA"].set_index("gene")
        sc = g[g.dataset == "scRNA"].set_index("gene")
        row = {"lineage": lin, "state": st}
        for ds, t in (("snRNA", sn), ("scRNA", sc)):
            row |= {f"{ds}_n_donors": int(t.n_donors.iloc[0]) if len(t) else 0,
                    f"{ds}_n_up": int((t.call == "up").sum()) if len(t) else 0,
                    f"{ds}_n_down": int((t.call == "down").sum()) if len(t) else 0}
        gi = sn.index.intersection(sc.index)
        if len(gi) > 50:
            row["lfc_spearman_sn_vs_sc"] = stats.spearmanr(sn.loc[gi, "log2fc"], sc.loc[gi, "log2fc"])[0]
            for dr in ("up", "down"):
                s1 = set(sn.index[sn.call == dr]) & set(gi)
                row[f"frac_{dr}_replicated_sc"] = (np.mean([np.sign(sc.at[x, "log2fc"]) == (1 if dr == "up" else -1)
                                                            and abs(sc.at[x, "log2fc"]) >= RS["program"]["rep_lfc"]
                                                            for x in s1]) if s1 else np.nan)
        for sub in ("REF", "CKD"):
            t = P[(P.dataset == "snRNA") & (P.lineage == lin) & (P.state == st) & (P.subset == sub)].set_index("gene")
            gi = sn.index.intersection(t.index)
            if len(gi) > 50:
                row[f"snRNA_lfc_spearman_all_vs_{sub}"] = stats.spearmanr(sn.loc[gi, "log2fc"], t.loc[gi, "log2fc"])[0]
                row[f"snRNA_n_donors_{sub}"] = int(t.n_donors.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def score_sets(P: pd.DataFrame) -> pd.DataFrame:
    a = P[P.subset == "all"]
    sn = a[(a.dataset == "snRNA") & a.in_score_set]
    sc = a[a.dataset == "scRNA"].set_index(["lineage", "state", "gene"])
    rep = []
    for _, r in sn.iterrows():
        k = (r.lineage, r.state, r.gene)
        v = sc.log2fc.get(k, np.nan)
        rep.append(bool(np.isfinite(v) and np.sign(v) == np.sign(r.log2fc) and abs(v) >= RS["program"]["rep_lfc"]))
    out = sn[["lineage", "state", "gene", "call", "log2fc", "d_z", "fdr", "n_donors", "mean_state", "mean_normal"]].copy()
    out["scRNA_log2fc"] = [sc.log2fc.get((r.lineage, r.state, r.gene), np.nan) for _, r in sn.iterrows()]
    out["replicated_scRNA"] = rep
    return out.rename(columns={"call": "direction"})


def main() -> None:
    set_global_seed(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    progs, fracs = [], []
    srcs = []
    for name in RS["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        srcs.append(pb.cdir / "source.json")
        cat = donor_categories(pb, name)
        progs.append(programs_for(pb, cat))
        f = state_fractions(pb)
        don = pb.donors
        f["dataset"] = name
        f["category"] = f.donor.map(cat)
        pc = CFG["shared_program"]["kpmp"]["procurement_col"][name]
        f["procurement"] = f.donor.map(don[pc].astype(str))
        f["egfr_bin"] = f.donor.map(don[EGFR].astype(str)) if EGFR in don else np.nan
        fracs.append(f)
        print(name, "done", flush=True)
    P = pd.concat(progs, ignore_index=True)
    S = summarise(P)
    Z = score_sets(P)
    F = pd.concat(fracs, ignore_index=True)
    outs = {"programs.tsv.gz": P, "program_summary.tsv": S, "program_sets.tsv": Z, "fractions.tsv.gz": F}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/programs", srcs, [O / f for f in outs], CFG["seed"],
                     {"program": RS["program"], "min_cells": RS["min_cells"], "lineages": RS["lineages"]})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S.round(3).to_string())
        print(Z.groupby(["lineage", "state", "direction"]).agg(n=("gene", "size"), rep=("replicated_scRNA", "mean")))


if __name__ == "__main__":
    main()
