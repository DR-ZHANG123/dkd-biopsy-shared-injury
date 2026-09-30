"""Published-signature panels of Fig. S12 (a-e). Plotting only; values are read by FigS12_signatures.py
from results/12_signature_audit/signature_metrics.tsv and manuscript/numbers/signature_{level,audit}.tsv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as fl
from matplotlib.lines import Line2D

UNITS = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "GSE96804-GPL17586", "ERCB_TUB_H1", "ERCB_TUB_H7", "GSE142025-RNAseq"]
SIG_COL = "#4C72B0"
AXIS_COL = "#C44E52"


def ulab(u: str) -> str:
    lab = fl.unit_label(u)
    return lab.replace("ERCB-", "ERCB\n") if lab.startswith("ERCB") else lab.replace("GSE", "GSE\n")


def published(sm: pd.DataFrame) -> pd.DataFrame:
    return sm[(sm.sig_id != "INJURY") & ~sm.sig_id.str.startswith("ORIENT") & sm.auc_dkd_ctrl.notna()]


def strip(ax, groups, col, s=2.5, alpha=0.55):
    for i, v in enumerate(groups):
        v = np.asarray(v)
        ax.scatter(i + fl.jitter(len(v), 0.3), v, s=s, lw=0, color=col, alpha=alpha, rasterized=True, zorder=2)
        ax.hlines(np.median(v), i - 0.36, i + 0.36, color="black", lw=0.9, zorder=3)


def panel_auc(ax, sm, kvs, name, pan):
    p = published(sm)
    inj = sm[sm.sig_id == "INJURY"].set_index("eval_unit").auc_dkd_ctrl
    groups = [p[p.eval_unit == u].auc_dkd_ctrl.values for u in UNITS]
    strip(ax, groups, SIG_COL)
    for i, u in enumerate(UNITS):
        ax.plot([i - 0.42, i + 0.42], [inj[u]] * 2, color=AXIS_COL, lw=1.4, zorder=4)
    ax.set_xticks(range(len(UNITS)), [ulab(u) for u in UNITS], fontsize=6)
    ax.set_xlim(-0.6, len(UNITS) - 0.4)
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("AUROC (DKD vs control)")
    frac = float(kvs["frac_pairs_auc_dkd_ctrl_le_injury"])
    n_pairs = int(float(kvs["n_signature_cohort_pairs"]))
    ax.text(0.02, 0.99, f"{frac * 100:.1f}% of {n_pairs:,} comparisons ≤ external score", transform=ax.transAxes,
            ha="left", va="top", fontsize=6)
    h = [Line2D([], [], marker="o", ls="", ms=2.5, color=SIG_COL, label="Published signature"),
         Line2D([], [], color=AXIS_COL, lw=1.4, label="External score"),
         Line2D([], [], color="black", lw=0.9, label="Median")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0, 0.935), fontsize=5.5, ncol=3, columnspacing=0.8)
    fl.save_source(name, pan, p[["sig_id", "compartment", "eval_unit", "n_genes_used", "auc_dkd_ctrl"]]
                   .merge(inj.rename("axis_score_auc_dkd_ctrl"), left_on="eval_unit", right_index=True))


def panel_pct(ax, sm, kvs, name, pan):
    p = published(sm)
    groups = [p[p.eval_unit == u].auc_dkd_ctrl_pct_vs_random.values for u in UNITS]
    strip(ax, groups, SIG_COL)
    ax.axhline(0.95, ls="--", color="black", lw=0.6)
    ax.set_xticks(range(len(UNITS)), [ulab(u) for u in UNITS], fontsize=6)
    ax.set_xlim(-0.6, len(UNITS) - 0.4)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Percentile in random null")
    frac = float(kvs["frac_pairs_auc_dkd_ctrl_not_above_random95"])
    ax.text(0.02, 0.99, f"{frac * 100:.1f}% not above 95th percentile", transform=ax.transAxes, ha="left",
            va="top", fontsize=6)
    fl.save_source(name, pan, p[["sig_id", "eval_unit", "auc_dkd_ctrl", "auc_dkd_ctrl_null_median",
                                 "auc_dkd_ctrl_pct_vs_random"]].assign(threshold=0.95))


def panel_adjusted(ax, sl, kvs, name, pan):
    d = sl.dropna(subset=["auc_dkd_other", "adj_auc_dkd_other"])
    jx = fl.jitter(len(d), 0.3)  # identical jitter to strip() for both columns (same n)
    for j, a, b in zip(jx, d.auc_dkd_other, d.adj_auc_dkd_other):
        ax.plot([j, 1 + j], [a, b], color="#9DB3D6", lw=0.3, alpha=0.6, zorder=1)
    strip(ax, [d.auc_dkd_other.values, d.adj_auc_dkd_other.values], SIG_COL, s=3)
    rn = [float(kvs["sig_median_of_cohort_mean_auc_dkd_other_null_median"]),
          float(kvs["sig_median_of_cohort_mean_adj_auc_dkd_other_null_median"])]
    md = [float(kvs["sig_median_of_cohort_mean_auc_dkd_other"]),
          float(kvs["sig_median_of_cohort_mean_adj_auc_dkd_other"])]
    for i in (0, 1):
        ax.hlines(rn[i], i - 0.36, i + 0.36, color="#8C8C8C", lw=0.9, ls=":", zorder=3)
        ax.text(i, 0.985, f"median {md[i]:.3f}\nrandom {rn[i]:.3f}", ha="center", va="top", fontsize=5.5)
    ax.axhline(0.5, ls="--", color="black", lw=0.5)
    ax.set_xticks([0, 1], ["Raw", "Injury–repair\nscore adjusted"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("AUROC (DKD vs other CKD)")
    h = [Line2D([], [], color="black", lw=0.9, label="Median"),
         Line2D([], [], color="#8C8C8C", lw=0.9, ls=":", label="Random median")]
    ax.legend(handles=h, loc="lower left", fontsize=5.5)
    fl.save_source(name, pan, d[["sig_id", "auc_dkd_other", "adj_auc_dkd_other", "auc_dkd_other_null_median",
                                 "adj_auc_dkd_other_null_median"]])


def panel_count(ax, sl, kvs, name, pan):
    cnt = sl.n_cohorts_adj_beats_random.value_counts().reindex(range(5), fill_value=0)
    ax.bar(cnt.index, cnt.values, color=SIG_COL, edgecolor="white", lw=0.3, width=0.7)
    for x, v in cnt.items():
        ax.text(x, v + 2, str(int(v)), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(5))
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, cnt.max() * 1.2)
    ax.set_xlabel("ERCB cohorts with adjusted\nAUROC > random 95th pct")
    ax.set_ylabel("Signatures")
    fl.save_source(name, pan, pd.DataFrame({"n_cohorts_beating_random95": cnt.index, "n_signatures": cnt.values}))


def panel_gse30122(ax, sl, kvs, name, pan):
    grp = [("yes", "Validated in\nGSE30122"), ("no", "No GSE30122\nvalidation")]
    vals = [sl[sl.gse30122 == g].adj_auc_dkd_other.dropna().values for g, _ in grp]
    strip(ax, vals, SIG_COL, s=4, alpha=0.7)
    for i, (g, _) in enumerate(grp):
        n = int((sl.gse30122 == g).sum())
        ax.text(i, 0.985, f"n = {n}\nmedian {float(kvs[f'gse30122_{g}_median_adj_auc_dkd_other']):.3f}",
                ha="center", va="top", fontsize=5.5)
    p = float(kvs["gse30122_yes_vs_no_adj_auc_dkd_other_MWU_p"])
    ax.text(0.5, 0.22, f"Mann–Whitney P = {p:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0.5, ls="--", color="black", lw=0.5)
    ax.set_xticks([0, 1], [lab for _, lab in grp])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("Adjusted AUROC\n(DKD vs other CKD)")
    fl.save_source(name, pan, sl[sl.gse30122.isin(["yes", "no"])][["sig_id", "gse30122", "adj_auc_dkd_other"]]
                   .assign(mwu_p_stored=p))
