"""Fig. S12 - single-cell directional agreement of the axis-adjusted DKD residual (panels a-b).

Sign agreement of the top residual genes (versus background genes) with the DKD-versus-control direction
inside each cell type, in GSE131882 and GSE209781; a glomerular residual, b tubulointerstitial residual.
Input (read only): results/13_celltype/sc_validation.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS12"
CT = {"PT": "PT", "PT_injured": "PT injured", "LOH": "LOH", "DCT": "DCT", "CNT_PC": "CNT/PC", "IC": "IC",
      "PODO": "Podocyte", "PEC": "PEC", "ENDO": "Endothelial", "ENDO_PT": "Peritub. endo.", "PERI": "Pericyte",
      "STROMA": "Stromal", "MAC": "Macrophage", "TCELL": "T cell", "BCELL": "B cell", "NK": "NK"}
DS = {"GSE131882": ("#4C72B0", -0.17), "GSE209781": ("#DD8452", 0.17)}


def panel(ax, sv, comp, key):
    s = sv[sv.compartment == comp].copy()
    order = sorted(s.cell_type.unique(), key=lambda c: CT.get(c, c))
    xpos = {c: i for i, c in enumerate(order)}
    for ds, (col, off) in DS.items():
        d = s[s.dataset == ds]
        x = d.cell_type.map(xpos) + off
        ax.vlines(x, d.sign_agreement_background, d.sign_agreement_top, color=col, lw=0.8)
        ax.scatter(x, d.sign_agreement_top, s=10, color=col, zorder=3)
        ax.scatter(x, d.sign_agreement_background, s=8, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
    ax.axhline(0.5, ls="--", color="black", lw=0.5)
    ax.set_xticks(range(len(order)), [CT.get(c, c) for c in order], rotation=90)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0.25, 0.78)
    ax.set_ylabel("Sign agreement with DKD")
    ax.set_title(f"{fl.COMP_LABEL[comp]}, adjusted DKD signal", fontsize=7, pad=7)
    fl.save_source(NAME, key, s.assign(x=s.cell_type.map(xpos)))


def build():
    sv = fl.read("results/13_celltype/sc_validation.tsv")
    fig = fl.new_fig(fl.FULL_W, 75 * fl.MM)
    gs = fig.add_gridspec(1, 2)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel(a, sv, "GLOM", "a")
    panel(b, sv, "TUB", "b")
    hs = [Line2D([], [], marker="o", ls="", ms=3, color=c, label=d) for d, (c, _) in DS.items()]
    hs.append(Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="#555555", label="Background genes"))
    b.legend(handles=hs, loc="upper right", fontsize=5.5, ncol=3, columnspacing=0.8)
    return fl.finalize(fig, {"a": a, "b": b})


if __name__ == "__main__":
    fl.run(build, NAME)
