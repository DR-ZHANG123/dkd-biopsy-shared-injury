"""Fig. S6 - plausibility of BayesPrism fractions and deconvolution features in development (panels a-c).

Inputs (read only): results/16_deconv/plausibility/all/proportions_by_batch.tsv (per-batch median fractions),
results/16_deconv/plausibility_key.tsv (BayesPrism vs NNLS key quantities, median/min/max over batches),
results/16_deconv/feature_set_comparison.tsv (development tasks, difference from B-L2, family ALL).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS6"
REF_COLORS = {"scRNA": "#4C72B0", "snRNA": "#DD8452"}
REF_LABEL = {"scRNA": "scRNA reference", "snRNA": "snRNA reference"}
METH_COLORS = {"BayesPrism": "#4C72B0", "NNLS": "#C44E52"}
KEY_Q = [("GLOM", "PODO", "Podocyte"), ("GLOM", "EC(all)", "Endothelial"), ("GLOM", "MES_VSMC", "Mesangial/VSMC"),
         ("GLOM", "PEC", "Parietal epithelial"), ("GLOM", "immune", "Immune"), ("TUB", "PT", "Proximal tubule"),
         ("TUB", "distal(TAL..IC)", "Distal nephron"), ("TUB", "immune", "Immune"), ("TUB", "FIB", "Fibroblast")]
FEAT_LABEL = {"B-L2+COMP": "B-L2 + composition", "B-L2+COMP+SF": "B-L2 + composition + state share",
              "B-L2|uc": "B-L2, cohort-centred", "D-COMP": "Composition", "D-COMP+RESID": "Composition + residual ranks",
              "D-COMP+SF": "Composition + state share", "D-COMP+SF+Z": "Composition + state share + cell-type ranks",
              "D-COMP+Z": "Composition + cell-type ranks", "D-COMP+Z|uc": "Composition + cell-type ranks, centred",
              "D-COMP|uc": "Composition, centred", "D-RANK_Z": "Bulk ranks, cell-type genes",
              "D-RESID": "Composition-residual ranks", "D-SF": "State share", "D-Z": "Cell-type ranks",
              "D-Z|uc": "Cell-type ranks, centred"}


def panel_a(axs):
    pb = fl.read("results/16_deconv/plausibility/all/proportions_by_batch.tsv")
    pb = pb[(pb.method == "BayesPrism") & (pb.subset == "all") & pb.compartment.isin(["GLOM", "TUB"])]
    out = []
    for ax, comp in zip(axs, ["GLOM", "TUB"]):
        d = pb[pb.compartment == comp]
        order = (d[d.reference == "scRNA"].groupby("type")["median"].median().sort_values(ascending=False).index)
        for k, ref in enumerate(["scRNA", "snRNA"]):
            s = d[d.reference == ref]
            for i, t in enumerate(order):
                v = s[s.type == t]["median"]
                x = i + (k - 0.5) * 0.36 + fl.jitter(len(v), 0.07)
                ax.scatter(x, v * 100, s=4, lw=0, color=REF_COLORS[ref], alpha=0.85, zorder=2)
                ax.hlines(v.median() * 100, i + (k - 0.5) * 0.36 - 0.15, i + (k - 0.5) * 0.36 + 0.15,
                          color="black", lw=0.7, zorder=3)
        ax.set_xticks(range(len(order)), [t.replace("_", "/") for t in order], rotation=90, fontsize=5.5)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_title(f"{fl.COMP_LABEL[comp]}, {d.series.nunique()} batches", fontsize=6.5, pad=7)
        out.append(d)
    axs[0].set_ylabel("BayesPrism fraction (%)")
    h = [Line2D([], [], marker="o", ls="", ms=3, color=c, label=REF_LABEL[r]) for r, c in REF_COLORS.items()]
    axs[1].legend(handles=h, loc="upper right", fontsize=5.5)
    fl.save_source(NAME, "a", pd.concat(out))


def panel_b(ax):
    pk = fl.read("results/16_deconv/plausibility_key.tsv")
    rows = []
    for comp, q, lab in KEY_Q:
        for meth in ["BayesPrism", "NNLS"]:
            r = pk[(pk.reference == "scRNA") & (pk.method == meth) & (pk.compartment == comp) & (pk.quantity == q)]
            if len(r):
                rows.append(r.iloc[0].to_dict() | {"label": f"{lab} ({'GLOM' if comp == 'GLOM' else 'TUB'})"})
    d = pd.DataFrame(rows)
    labs = list(dict.fromkeys(d.label))
    for k, meth in enumerate(["BayesPrism", "NNLS"]):
        s = d[d.method == meth].set_index("label").reindex(labs)
        y = np.arange(len(labs))[::-1] + (0.18 if k == 0 else -0.18)
        ax.errorbar(s.median_of_batches * 100, y, xerr=[(s.median_of_batches - s.min_batch) * 100,
                                                        (s.max_batch - s.median_of_batches) * 100],
                    fmt="o", ms=3, color=METH_COLORS[meth], lw=0.7, capsize=0, label=meth)
    ax.set_yticks(np.arange(len(labs))[::-1], labs, fontsize=5.8)
    ax.set_xlim(-2, 100)
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_xlabel("Fraction, median over batches (range) (%)")
    ax.legend(loc="lower right", fontsize=5.5, title="scRNA reference", title_fontsize=5.5)
    fl.save_source(NAME, "b", d)


def panel_c(ax):
    fc = fl.read("results/16_deconv/feature_set_comparison.tsv")
    d = fc[fc.family == "ALL"].copy()
    d["label"] = d.method.map(FEAT_LABEL).fillna(d.method)
    order = (d[d.reference == "scRNA"].sort_values("d_adj_auroc_mean").label.tolist())
    order += [x for x in d[d.reference == "snRNA"].sort_values("d_adj_auroc_mean").label if x not in order]
    ypos = {lab: i for i, lab in enumerate(order[::-1])}
    for k, ref in enumerate(["scRNA", "snRNA"]):
        s = d[d.reference == ref]
        y = s.label.map(ypos) + (0.17 if k == 0 else -0.17)
        ax.errorbar(s.d_adj_auroc_mean, y, xerr=[s.d_adj_auroc_mean - s.d_adj_auroc_lo,
                                                 s.d_adj_auroc_hi - s.d_adj_auroc_mean],
                    fmt="o", ms=2.8, color=REF_COLORS[ref], lw=0.7, capsize=0, label=REF_LABEL[ref])
    ax.axvline(0, ls="--", color="black", lw=0.6)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()), fontsize=5.5)
    ax.set_ylim(-0.7, len(ypos) - 0.3)
    ax.set_xlim(-0.19, 0.05)
    ax.set_xlabel("Δ adjusted AUROC vs B-L2 (95% CI)")
    n = int(d.n_cells.iloc[0])
    ax.legend(loc="lower left", fontsize=5.5, title=f"{n} development cells", title_fontsize=5.5)
    fl.save_source(NAME, "c", d)


def build():
    fig = fl.new_fig(fl.FULL_W, 150 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[0.8, 1.2])
    a1, a2 = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1, 1.15])
    b, c = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    for ax in (b, c):
        fl.reserve_label(ax)
    panel_a([a1, a2])
    panel_b(b)
    panel_c(c)
    return fl.finalize(fig, {"a": a1, "b": b, "c": c})


if __name__ == "__main__":
    fl.run(build, NAME)
