"""Fig. S7 - KPMP donor-level decomposition details (panels a-d).

a  cell-type abundance change, CKD vs reference donors (Hedges g, 95% CI), snRNA and scRNA
b  per-cell-type state and composition contributions to the SCP difference (snRNA), all donors and
   percutaneous-only comparison, glomerular and tubulointerstitial cores
c  core-gene classes (state / composition / ...) by compartment and dataset
d  within-cell-type direction agreement of core genes vs background (snRNA)
Inputs (read only): results/19_shared_program/figdata/fig2_kpmp_{abundance,decomposition,gene_class,
set_direction}.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "FigS7"
FD = "results/19_shared_program/figdata/"
CT = {"PT": "PT", "PT_injured": "PT injured", "LOH": "LOH", "DCT": "DCT", "CNT_PC": "CNT/PC", "IC": "IC",
      "PODO": "Podocyte", "PEC": "PEC", "MES": "Mesangial", "ENDO": "Endothelial", "ENDO_PT": "Peritub. endo.",
      "PERI": "Pericyte", "STROMA": "Stromal", "MAC": "Macrophage", "DC": "Dendritic", "TCELL": "T cell",
      "BCELL": "B cell", "NK": "NK", "MAST": "Mast", "NEUTRO": "Neutrophil"}
DS_COL = {"snRNA": "#4C72B0", "scRNA": "#DD8452"}


def panel_a(ax):
    ab = fl.read(FD + "fig2_kpmp_abundance.tsv")
    ab = ab[ab.contrast == "CKD_vs_REF"].copy()
    order = (ab[ab.dataset == "snRNA"].sort_values("hedges_g").cell_type.tolist())
    order += [c for c in ab.cell_type.unique() if c not in order]
    ypos = {c: i for i, c in enumerate(order)}
    for k, (ds, off) in enumerate((("snRNA", 0.17), ("scRNA", -0.17))):
        d = ab[ab.dataset == ds]
        y = d.cell_type.map(ypos) + off
        ax.errorbar(d.hedges_g, y, xerr=[d.hedges_g - d.g_ci_lo, d.g_ci_hi - d.hedges_g], fmt="o", ms=2.5,
                    color=DS_COL[ds], elinewidth=0.6, capsize=0, label=f"KPMP {ds}")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)), [CT.get(c, c) for c in order], fontsize=5.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Abundance change (Hedges g)")
    ax.legend(loc="lower right", fontsize=5.5)
    fl.save_source(NAME, "a", ab.assign(y=ab.cell_type.map(ypos)))


def panel_b(axs):
    dc = fl.read(FD + "fig2_kpmp_decomposition.tsv")
    d = dc[(dc.dataset == "snRNA") & dc.kind.isin(["state_by_type", "comp_by_type"])].copy()
    top = (d[d.contrast == "CKD_vs_REF"].assign(a=lambda x: x.delta.abs()).groupby("cell_type").a.max()
           .sort_values(ascending=False).index[:10].tolist())
    d = d[d.cell_type.isin(top)]
    out = []
    styles = {("state_by_type", "CKD_vs_REF"): ("#C44E52", 1.0), ("state_by_type", "CKD_perc_vs_REF_perc"):
              ("#C44E52", 0.45), ("comp_by_type", "CKD_vs_REF"): ("#4C72B0", 1.0),
              ("comp_by_type", "CKD_perc_vs_REF_perc"): ("#4C72B0", 0.45)}
    offs = {("state_by_type", "CKD_vs_REF"): 0.3, ("state_by_type", "CKD_perc_vs_REF_perc"): 0.1,
            ("comp_by_type", "CKD_vs_REF"): -0.1, ("comp_by_type", "CKD_perc_vs_REF_perc"): -0.3}
    for ax, comp in zip(axs, ["GLOM", "TUB"]):
        x = d[d.compartment == comp]
        order = (x[(x.contrast == "CKD_vs_REF")].groupby("cell_type").delta.sum().reindex(top).fillna(0)
                 .sort_values().index.tolist())
        ypos = {c: i for i, c in enumerate(order)}
        for key, (col, alpha) in styles.items():
            s = x[(x.kind == key[0]) & (x.contrast == key[1])]
            ax.barh(s.cell_type.map(ypos) + offs[key], s.delta, height=0.2, color=col, alpha=alpha, lw=0)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(order)), [CT.get(c, c) for c in order], fontsize=5.5)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(fl.COMP_LABEL[comp], fontsize=6.5, pad=3)
        ax.set_xlabel("Δ injury–repair score")
        lim = x.delta.abs().max() * 1.15
        ax.set_xlim(-0.25 * lim, lim)
        out.append(x.assign(y=x.cell_type.map(ypos)))
    hs = [Patch(color="#C44E52", label="Within-lineage, all donors"), Patch(color="#C44E52", alpha=0.45,
          label="Within-lineage, percutaneous"), Patch(color="#4C72B0", label="Compositional, all donors"),
          Patch(color="#4C72B0", alpha=0.45, label="Compositional, percutaneous")]
    axs[1].legend(handles=hs, loc="lower right", fontsize=5)
    fl.save_source(NAME, "b", pd.concat(out)[["compartment", "contrast", "kind", "cell_type", "n_A", "n_B",
                                               "delta", "p_perm", "y"]])


def panel_c(ax):
    gc = fl.read(FD + "fig2_kpmp_gene_class.tsv")
    gc = gc[gc.contrast == "CKD_vs_REF"]
    t = gc.groupby(["compartment", "dataset", "class"]).n_genes.sum().unstack(fill_value=0)
    classes = ["state", "state+composition", "composition", "opposite_state", "unresolved"]
    cols = {"state": "#C44E52", "state+composition": "#8172B3", "composition": "#4C72B0",
            "opposite_state": "#CCB974", "unresolved": "#CCCCCC"}
    labs = {"state": "Within-lineage", "state+composition": "Both", "composition": "Compositional",
            "opposite_state": "Opposite, within-lineage", "unresolved": "Unresolved"}
    idx = [("GLOM", "snRNA"), ("GLOM", "scRNA"), ("TUB", "snRNA"), ("TUB", "scRNA")]
    t = t.reindex(idx)
    left = np.zeros(len(idx))
    for c in classes:
        v = t[c].values
        ax.barh(range(len(idx)), v, left=left, color=cols[c], edgecolor="white", lw=0.3, height=0.65,
                label=labs[c])
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 0.12 * t.loc[idx[i]].sum() and c != "unresolved":
                ax.text(l + w / 2, i, f"{w}", ha="center", va="center", fontsize=5, color="white")
        left += v
    for i, key in enumerate(idx):
        ax.text(left[i] * 1.0 + 8, i, f"prop. {t.loc[key, 'composition']}", ha="left", va="center", fontsize=5)
    ax.set_yticks(range(len(idx)), [f"{'GLOM' if c == 'GLOM' else 'TUB'} {d}" for c, d in idx], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.2)
    ax.set_xlabel("Response genes")
    ax.set_ylim(len(idx) + 1.6, -0.6)
    ax.legend(loc="lower center", ncol=3, fontsize=5, bbox_to_anchor=(0.5, 0.0), columnspacing=0.8)
    fl.save_source(NAME, "c", t.reset_index())


def panel_d(ax):
    sd = fl.read(FD + "fig2_kpmp_set_direction.tsv")
    sd = sd[(sd.dataset == "snRNA") & (sd.contrast == "CKD_vs_REF")].copy()
    order = sd[sd.compartment == "TUB"].sort_values("agree_top").cell_type.tolist()
    xpos = {c: i for i, c in enumerate(order)}
    for comp, off in (("GLOM", -0.17), ("TUB", 0.17)):
        s = sd[sd.compartment == comp]
        x = s.cell_type.map(xpos) + off
        ax.vlines(x, s.agree_background, s.agree_top, color=fl.COMP_COLORS[comp], lw=0.8)
        ax.scatter(x, s.agree_top, s=9, color=fl.COMP_COLORS[comp], zorder=3,
                   edgecolor=np.where(s.p_perm < 0.05, "black", "none"), lw=0.4)
        ax.scatter(x, s.agree_background, s=7, facecolor="white", edgecolor=fl.COMP_COLORS[comp], lw=0.6,
                   zorder=3)
    ax.axhline(0.5, ls="--", color="black", lw=0.5)
    ax.set_xticks(range(len(order)), [CT.get(c, c) for c in order], rotation=90, fontsize=5.5)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("Direction agreement")
    hs = [Line2D([], [], marker="o", ls="", ms=3, color=fl.COMP_COLORS["GLOM"], label="GLOM response genes"),
          Line2D([], [], marker="o", ls="", ms=3, color=fl.COMP_COLORS["TUB"], label="TUB response genes"),
          Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="#555555", label="Background"),
          Line2D([], [], marker="o", ls="", ms=3, mfc="#AAAAAA", mec="black", label="P < 0.05")]
    ax.legend(handles=hs, loc="upper left", ncol=4, fontsize=5, columnspacing=0.8)
    fl.save_source(NAME, "d", sd.assign(x=sd.cell_type.map(xpos)))


def build():
    fig = fl.new_fig(fl.FULL_W, 140 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    r1 = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 1])
    a, b1, b2 = fig.add_subplot(r1[0]), fig.add_subplot(r1[1]), fig.add_subplot(r1[2])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[0.8, 1.2])
    c, d = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    for ax in (a, c, d):
        fl.reserve_label(ax)
    panel_a(a)
    panel_b([b1, b2])
    panel_c(c)
    panel_d(d)
    return fl.finalize(fig, {"a": a, "b": b1, "c": c, "d": d})


if __name__ == "__main__":
    fl.run(build, NAME)
