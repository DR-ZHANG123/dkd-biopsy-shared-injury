"""Fig. S5 - stability of the core genes (panels a-c) and pathway annotation of the response (panel d).

Inputs (read only): results/19_shared_program/core/stability.tsv,
results/19_shared_program/figdata/fig1_core_forest.tsv (representative core genes, per-source effects).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
import fig_extra_panels as xp  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS5"
N_PER_DIR = 8
SRC_MARK = ["o", "s", "^", "v"]


def panel_a(ax, st):
    lo = st[st.kind == "leave_one_out"].reset_index(drop=True)
    order = []
    ypos, yy = [], 0.0
    for comp in ["GLOM", "TUB"]:
        s = lo[lo.compartment == comp]
        for _, r in s.iterrows():
            ypos.append(yy)
            order.append(r)
            yy -= 1
        yy -= 0.6
    lo2 = pd.DataFrame(order).assign(y=ypos)
    ax.barh(lo2.y, lo2.frac_full_core_retained * 100, color=[fl.COMP_COLORS[c] for c in lo2.compartment],
            height=0.65, edgecolor="white", lw=0.3)
    for r in lo2.itertuples():
        ax.text(r.frac_full_core_retained * 100 + 1.5, r.y, f"{r.frac_full_core_retained * 100:.0f}%  "
                f"(ρ = {r.spearman_g:.2f})", va="center", fontsize=5.5)
    ax.set_yticks(lo2.y, [f"without {fl.unit_label(a)}" for a in lo2.a], fontsize=6)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Response genes retained (%)")
    ax.set_ylim(min(ypos) - 0.7, 0.7)
    fl.save_source(NAME, "a", lo2)


def panel_b(ax, st):
    sp = st[st.kind == "source_pair"].reset_index(drop=True)
    ypos, yy = [], 0.0
    for comp in ["GLOM", "TUB"]:
        for _ in range((sp.compartment == comp).sum()):
            ypos.append(yy)
            yy -= 1
        yy -= 0.6
    sp = sp.sort_values("compartment", kind="stable").assign(y=ypos)
    ax.barh(sp.y, sp.spearman_g, color=[fl.COMP_COLORS[c] for c in sp.compartment], height=0.65,
            edgecolor="white", lw=0.3)
    for r in sp.itertuples():
        ax.text(max(r.spearman_g, 0) + 0.01, r.y, f"{r.spearman_g:.2f}", va="center", fontsize=5.5)
    ax.set_yticks(sp.y, [f"{fl.unit_label(a)} vs {fl.unit_label(b)}" for a, b in zip(sp.a, sp.b)], fontsize=5.5)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(min(ypos) - 0.7, 0.7)
    ax.set_xlabel("Spearman ρ of gene effects")
    h = [Line2D([], [], marker="s", ls="", ms=4, color=fl.COMP_COLORS[c], label=fl.COMP_LABEL[c])
         for c in ["GLOM", "TUB"]]
    ax.legend(handles=h, loc="lower right", fontsize=5.5)
    fl.save_source(NAME, "b", sp)


def panel_c(axs, fo):
    out = []
    for ax, comp in zip(axs, ["GLOM", "TUB"]):
        f = fo[fo.compartment == comp]
        genes = (f.drop_duplicates("gene").assign(a=lambda x: x.g_re.abs())
                 .sort_values("a", ascending=False).groupby("direction").head(N_PER_DIR)
                 .sort_values("g_re"))
        srcs = list(dict.fromkeys(f.source))
        for yi, g in enumerate(genes.itertuples()):
            ax.errorbar(g.g_re, yi, xerr=1.96 * g.se, fmt="D", ms=3.5, color="black", lw=0.7, capsize=0,
                        zorder=3)
            s = f[f.gene == g.gene]
            for k, src in enumerate(srcs):
                v = s[s.source == src].g_source
                if len(v):
                    ax.scatter(v.values, [yi + 0.18 * (k - (len(srcs) - 1) / 2)] * len(v), s=6,
                               marker=SRC_MARK[k], color=fl.PALETTE[k + 4 if comp == "TUB" else k], lw=0,
                               alpha=0.9, zorder=2)
        ax.axvline(0, color="#999999", lw=0.5, zorder=0)
        ax.set_yticks(range(len(genes)), genes.gene)
        for t in ax.get_yticklabels():
            t.set_fontstyle("italic")
        ax.set_ylim(-0.7, len(genes) - 0.3)
        ax.set_xlabel("Effect (g), non-DKD vs control")
        ax.set_title(fl.COMP_LABEL[comp], fontsize=6.5, pad=4)
        h = [Line2D([], [], marker=SRC_MARK[k], ls="", ms=3, color=fl.PALETTE[k + 4 if comp == "TUB" else k],
                    label=fl.unit_label(src)) for k, src in enumerate(srcs)]
        h.append(Line2D([], [], marker="D", ls="-", ms=3, color="black", lw=0.7, label="Pooled (95% CI)"))
        ax.legend(handles=h, loc="lower right", fontsize=5.2)
        out.append(f[f.gene.isin(genes.gene)])
    fl.save_source(NAME, "c", pd.concat(out))


def build():
    st = fl.read("results/19_shared_program/core/stability.tsv")
    fo = fl.read("results/19_shared_program/figdata/fig1_core_forest.tsv")
    fig = fl.new_fig(fl.FULL_W, 222 * fl.MM)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.75, 1.45, 1.75])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[1, 1])
    a, b = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 2)
    c1, c2 = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    r3 = gs[2].subgridspec(1, 3, width_ratios=[0.35, 1, 0.35])
    d = fig.add_subplot(r3[1])
    for ax in (a, b, d):
        fl.reserve_label(ax)
    panel_a(a, st)
    panel_b(b, st)
    panel_c([c1, c2], fo)
    xp.pathways(d, NAME, "d")
    return fl.finalize(fig, {"a": a, "b": b, "c": c1, "d": d})


if __name__ == "__main__":
    fl.run(build, NAME)
