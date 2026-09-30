"""Fig. S11 - one-time DKD evaluation of the classifier trained on the adjusted signal (panels a-b).

a  leave-one-cohort-out DKD evaluation in ERCB: adjusted AUROC of the classifier and all comparators, with
   paired bootstrap differences; b  pre-specified independent cohorts (raw AUROC, paired bootstrap vs B-L2).
The development ablations, fold stability and key-gene panels are no longer shown (tables in Additional file 5).
Inputs (read only): results/15_model/final_dkd/{metrics,paired_bootstrap}.tsv,
results/17_independent/{metrics,paired_bootstrap_deltas}.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS11"
METHODS = ["RRG-ID", "B-L2", "B2-PCA", "B-cPCA", "B-rankLASSO", "B-L2-unitcenter"]
ERCB = ["ERCB_GLOM_H1", "ERCB_GLOM_H7", "ERCB_TUB_H1", "ERCB_TUB_H7"]
COHORTS = ["GSE162830", "KPMP", "GSE166239"]
COHORT_LABEL = {"GSE162830": "GSE162830", "KPMP": "KPMP sections", "GSE166239": "GSE166239"}
NEG_LABEL = {"ING": "ING", "HT+OTHER": "other CKD", "HT": "HT", "CONTROL": "healthy"}


def mlabel(m: str) -> str:
    return fl.METHOD_LABEL.get(m, m)


def panel_a(ax, axd):
    m = fl.read("results/15_model/final_dkd/metrics.tsv")
    bs = fl.read("results/15_model/final_dkd/paired_bootstrap.tsv")
    m = m[m.method.isin(METHODS)]
    w = 0.8 / len(METHODS)
    for k, meth in enumerate(METHODS):
        d = m[m.method == meth].set_index("test_unit").loc[ERCB]
        x = np.arange(len(ERCB)) - 0.4 + w * (k + 0.5)
        ax.bar(x, d.adj_auroc - 0.5, bottom=0.5, width=w * 0.92, color=fl.METHOD_COLORS[meth], edgecolor="white",
               lw=0.3, label=mlabel(meth))
    ax.axhline(0.5, color="black", lw=0.6)
    ax.text(-0.45, 0.37, "External score: 0.5 by construction", ha="left", va="bottom", fontsize=5.5)
    lab = [f"{fl.unit_label(u).replace('ERCB-', '')}\n{int(m[m.test_unit == u].n_dkd.iloc[0])}/{int(m[m.test_unit == u].n_other.iloc[0])}"
           for u in ERCB]
    ax.set_xticks(range(len(ERCB)), lab, fontsize=5.5)
    ax.set_ylim(0.35, 1.08)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("Adjusted AUROC")
    ax.legend(ncol=3, loc="upper left", fontsize=5.2, columnspacing=0.6, handlelength=0.9)
    b = bs[(bs.metric == "adj_auroc") & bs.comparator.isin(METHODS[1:])].set_index("comparator").loc[METHODS[1:]]
    b = b.reset_index()
    y = np.arange(len(b))[::-1]
    axd.errorbar(b.delta, y, xerr=[b.delta - b.lo, b.hi - b.delta], fmt="o", ms=3, color="#C44E52",
                 ecolor="#C44E52", elinewidth=0.8, capsize=1.5)
    axd.axvline(0, ls="--", color="black", lw=0.5)
    axd.set_yticks(y, [mlabel(c) for c in b.comparator], fontsize=5.5)
    axd.set_xlabel("Δ adj. AUROC\n(classifier − other)")
    axd.set_xlim(-0.12, 0.36)
    axd.set_ylim(-0.6, len(b) - 0.4)
    fl.save_source(NAME, "a", m.assign(panel="auroc"))
    fl.save_source(NAME, "a_bootstrap", b)


def panel_b(ax, axd):
    m = fl.read("results/17_independent/metrics.tsv")
    bs = fl.read("results/17_independent/paired_bootstrap_deltas.tsv")
    rows = []
    for c in COHORTS:
        for ct in ("primary", "secondary"):
            rows.append((c, ct))
    y = np.arange(len(rows))[::-1]
    labels, src, dl = [], [], []
    for yi, (c, ct) in zip(y, rows):
        d = m[(m.cohort == c) & (m.contrast == ct)]
        r = d[d.method == "RRG-ID"].iloc[0]
        for meth, off, mk in (("B-L2", -0.18, "s"), ("B2-PCA", 0.18, "^")):
            q = d[d.method == meth].iloc[0]
            ax.scatter(q.auroc, yi + off, marker=mk, s=10, color=fl.METHOD_COLORS[meth], zorder=2)
        ax.errorbar(r.auroc, yi, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="o", ms=3.2,
                    color=fl.METHOD_COLORS["RRG-ID"], elinewidth=0.8, capsize=1.5, zorder=3)
        labels.append(f"{COHORT_LABEL[c]}\nDKD {r.n_dkd} vs {NEG_LABEL[r.negatives]} {r.n_neg}")
        src.append(d)
        b = bs[(bs.cohort == c) & (bs.contrast == ct) & (bs.method == "RRG-ID") & (bs.vs == "B-L2")
               & (bs.metric == "raw")].iloc[0]
        axd.errorbar(b.delta, yi, xerr=[[b.delta - b.lo], [b.hi - b.delta]], fmt="o", ms=3, color="#C44E52",
                     elinewidth=0.8, capsize=1.5)
        dl.append(b)
    for yy in (1.5, 3.5):
        ax.axhline(yy, color="#DDDDDD", lw=0.5)
        axd.axhline(yy, color="#DDDDDD", lw=0.5)
    ax.axvline(0.5, ls="--", color="black", lw=0.5)
    axd.axvline(0, ls="--", color="black", lw=0.5)
    ax.set_yticks(y, labels, fontsize=5.3)
    axd.set_yticks(y, [""] * len(y))
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    axd.set_ylim(-0.6, len(rows) - 0.4)
    axd.set_xlim(-0.65, 0.4)
    ax.set_xlabel("AUROC (raw)")
    axd.set_xlabel("Δ AUROC vs B-L2")
    h = [Line2D([], [], marker="o", ls="", ms=3, color=fl.METHOD_COLORS["RRG-ID"], label="Classifier (95% CI)"),
         Line2D([], [], marker="s", ls="", ms=3, color=fl.METHOD_COLORS["B-L2"], label=fl.METHOD_LABEL["B-L2"]),
         Line2D([], [], marker="^", ls="", ms=3, color=fl.METHOD_COLORS["B2-PCA"], label=fl.METHOD_LABEL["B2-PCA"])]
    leg = ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.3,
                    frameon=False, columnspacing=0.6, handletextpad=0.2)
    leg.set_gid("free")
    fl.save_source(NAME, "b", pd.concat(src))
    fl.save_source(NAME, "b_bootstrap", pd.DataFrame(dl))


def build():
    fig = fl.new_fig(fl.FULL_W, 115 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1.6, 0.8])
    a, a2 = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1.6, 0.8])
    b, b2 = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    fl.reserve_label(a)
    panel_a(a, a2)
    panel_b(b, b2)
    return fl.finalize(fig, {"a": a, "b": b})


if __name__ == "__main__":
    fl.run(build, NAME)
