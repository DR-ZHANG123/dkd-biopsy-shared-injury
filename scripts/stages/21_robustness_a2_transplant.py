"""Stage 21 A2：与移植肾 injury–repair 基因集（Halloran/Famulski，ATAGC）的直接比较。

基因集（results/21_robustness/A2_transplant/genesets，来源与 md5 见 sources.tsv）：IRRAT394（JASN 2012 Suppl. Table 1，AKI 上调）、
IRRAT30（= AKIT30，前 30）、KT1 / KT2（肾实质转录本，损伤后下降）、IRITD3 / IRITD5（小鼠同种同基因移植损伤诱导，人源化）、
ENDAT（内皮；非损伤集，作对照）。
(1) 重叠：基因宇宙 = 该区室 stage 19 meta 基因（全部来源都测到）；上调集 vs 反应基因上调、下调集 vs 反应基因下调的超几何检验
    （OR、P）；方向一致率 = 基因集中被测到的基因在 stage 19 合并效应 g_re 上与集定义同号的比例（及在反应基因内的同号比例）。
(2) 分数：IRRAT394 / IRRAT30 / IRITD3（只上调，平均秩）、IRRAT394−KT1（带符号）、ENDAT；本文反应分数（带符号；只上调版本同口径对照）。
    队列 = stage 11 评估单元（反应基因用去掉本单元后重估的集合）+ stage 19d 独立队列。每队列：两类分数的 Spearman（全体 / 病人内）、
    疾病 vs 对照 AUROC，以及相互调整后的 AUROC（Janes–Pepe：本文分数调整 IRRAT 分数，反之亦然）。
产出 results/21_robustness/A2_transplant/{overlap.tsv, cohort_scores.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import adjusted_auroc, full_ranks  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, units  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core, loo_sets  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

O = OUT / "A2_transplant"
GS = O / "genesets"
SETS = {"IRRAT394": "up", "IRRAT30": "up", "IRITD3": "up", "IRITD5": "up", "KT1": "down", "KT2": "down", "ENDAT": "up"}


def load_sets() -> dict[str, list[str]]:
    return {k: sorted(set(pd.read_csv(GS / f"{k}.tsv", sep="\t").gene_symbol.dropna().astype(str))) for k in SETS}


def overlap_rows(sets) -> list[dict]:
    rows = []
    for comp in ("GLOM", "TUB"):
        M = pd.read_csv(SCP_OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t").set_index("gene")
        uni = M.index
        up, dn = load_core(comp)
        for k, genes in sets.items():
            g = uni.intersection(genes)
            tgt = set(up) if SETS[k] == "up" else set(dn)
            a = len(set(g) & tgt)
            b, c = len(g) - a, len(tgt) - a
            d = len(uni) - a - b - c
            orr, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            sgn = 1 if SETS[k] == "up" else -1
            opp = set(dn) if SETS[k] == "up" else set(up)
            rows.append({"compartment": comp, "gene_set": k, "set_direction": SETS[k], "n_set": len(genes), "n_measured": len(g),
                         "n_response_same_direction": len(tgt), "n_overlap_same_direction": a,
                         "n_overlap_opposite_direction": len(set(g) & opp),
                         "frac_set_in_response_same_dir": a / max(len(g), 1),
                         "expected_frac": len(tgt) / len(uni), "odds_ratio": orr, "p_hypergeom": p,
                         "frac_set_sign_concordant_g_re": float((np.sign(M.g_re[g]) == sgn).mean()),
                         "median_g_re_set": float(M.g_re[g].median()),
                         "overlap_genes": ";".join(sorted(set(g) & tgt)[:60])})
    return rows


def set_score(R: pd.DataFrame, up: list[str], dn: list[str] | None = None) -> np.ndarray:
    u = [x for x in up if x in R.columns]
    s = R[u].mean(1).to_numpy()
    if dn:
        s = s - R[[x for x in dn if x in R.columns]].mean(1).to_numpy()
    return s


def cohort_rows(name, R, d, pos, core_up, core_dn, sets) -> list[dict]:
    defs = {"response_signed": (core_up, core_dn), "response_up_only": (core_up, None), "IRRAT394": (sets["IRRAT394"], None),
            "IRRAT30": (sets["IRRAT30"], None), "IRITD3": (sets["IRITD3"], None),
            "IRRAT394_minus_KT1": (sets["IRRAT394"], sets["KT1"]), "ENDAT": (sets["ENDAT"], None)}
    sc = {k: set_score(R, u, dn) for k, (u, dn) in defs.items()}
    nmeas = {k: sum(x in R.columns for x in u) + (sum(x in R.columns for x in dn) if dn else 0) for k, (u, dn) in defs.items()}
    m = np.isin(d, pos + ["CONTROL"])
    y = np.isin(d[m], pos).astype(int)
    pat = d != "CONTROL"
    rows = []
    for k, v in sc.items():
        r = {"cohort": name, "positive": "+".join(pos), "n_pos": int(y.sum()), "n_ctrl": int((1 - y).sum()), "score": k,
             "n_genes_measured": int(nmeas[k]),
             "auroc": float(auc_rows(y, v[m][None])[0]) if y.sum() and (1 - y).sum() else np.nan,
             "spearman_vs_response_all": stats.spearmanr(v, sc["response_signed"])[0],
             "spearman_vs_response_patients": stats.spearmanr(v[pat], sc["response_signed"][pat])[0] if pat.sum() > 5 else np.nan}
        if k.startswith(("IRRAT", "IRITD")) and y.sum() >= 3 and (1 - y).sum() >= 3:
            r["auroc_response_adjusted_for_this"] = adjusted_auroc(y, sc["response_signed"][m], v[m])
            r["auroc_this_adjusted_for_response"] = adjusted_auroc(y, v[m], sc["response_signed"][m])
        rows.append(r)
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    sets = load_sets()
    ov = pd.DataFrame(overlap_rows(sets))
    rows = []
    for comp, spec in CFG["injury"]["compartments"].items():
        for u in spec["eval"]:
            R = full_ranks(U[u], s).dropna(axis=1)
            d = s.loc[R.index, "diagnosis"].to_numpy()
            up, dn = loo_sets(comp, u, s, U)
            rows += [r | {"compartment": comp, "cohort_role": "evaluation_DKD"} for r in cohort_rows(u, R, d, ["DKD"], up, dn, sets)]
            others = sorted(set(d) - {"CONTROL", "DKD", "OTHER", "UNKNOWN", "TMD"})
            if others:
                rows += [r | {"compartment": comp, "cohort_role": "evaluation_nonDKD"}
                         for r in cohort_rows(u, R, d, others, up, dn, sets)]
    S19 = import_module("19_shared_program_replicate")
    for name, spec in CFG["shared_program"]["replicate"]["cohorts"].items():
        R, meta = S19.load_cohort(name, spec, s, U)
        up, dn = load_core(spec["core"])
        rows += [r | {"compartment": spec["core"], "cohort_role": "independent"}
                 for r in cohort_rows(name, R, meta.diagnosis.to_numpy(), spec["positive"], up, dn, sets)]
    C = pd.DataFrame(rows)
    ov.to_csv(O / "overlap.tsv", sep="\t", index=False)
    C.to_csv(O / "cohort_scores.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A2_transplant", [GS / f"{k}.tsv" for k in SETS] + [GS / "sources.tsv"],
                     [O / "overlap.tsv", O / "cohort_scores.tsv"], CFG["seed"], {"sets": SETS})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(ov.drop(columns="overlap_genes").round(3).to_string())
        print(C.pivot_table(index=["cohort_role", "cohort"], columns="score", values="auroc").round(3).to_string())
        print(C.pivot_table(index=["cohort_role", "cohort"], columns="score", values="spearman_vs_response_patients").round(2).to_string())
        print(C[C.score.isin(["IRRAT394", "IRRAT394_minus_KT1"])][["cohort", "score", "auroc", "auroc_response_adjusted_for_this",
                                                                    "auroc_this_adjusted_for_response"]].round(3).to_string())


if __name__ == "__main__":
    main()
