"""Fig. S14 - recovery of known disease-specific signals added to real ERCB patients after adjustment for the
injury–repair score (panels a-c).

Input (read only): results/21_robustness/A4_injection/summary.tsv (100 repeats per scenario).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS14"
UNITS = [("ERCB_GLOM_H1", "ERCB-GLOM-H1"), ("ERCB_TUB_H1", "ERCB-TUB-H1")]
GROUPS = [("random", "Random patients (20)"), ("sev_dkd", "Severity-linked (DKD-like)"),
          ("sev_strong", "Severity-linked (strong)"), ("real_DKD", "Real DKD patients")]
GCOL = dict(zip([g for g, _ in GROUPS], ["#8C8C8C", "#DD8452", "#C44E52", "#4C72B0"]))
GENES = [("random_noncore", "o", "Random genes"), ("score_corr_noncore", "s", "Score-correlated genes"),
         ("response_up", "^", "Response genes")]


def panel_a(ax, s):
    rows = []
    for ui, (u, _) in enumerate(UNITS):
        for gi, (g, _) in enumerate(GROUPS):
            for ki, (gm, mk, _) in enumerate(GENES):
                d = s[(s.unit == u) & (s.group_mode == g) & (s.gene_mode == gm)]
                x = ui * 5 + gi + (ki - 1) * 0.22
                ax.scatter(np.full(len(d), x), d.recovery_mean_scale * 100, s=6, marker=mk, color=GCOL[g], lw=0,
                           alpha=0.8)
                rows.append(d.assign(x=x))
    ax.axhline(100, color="black", lw=0.5, ls="--")
    ax.set_xticks([1.5, 6.5], [lab for _, lab in UNITS])
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(60, 104)
    ax.set_ylabel("Injected signal retained\nafter adjustment (%)")
    fl.save_source(NAME, "a", pd.concat(rows))


def panel_b(ax, s):
    d = s[(s.gene_mode == "random_noncore") & (s.effect_mode == "constant")]
    rows = []
    for u, lab in UNITS:
        for g, _ in GROUPS:
            t = d[(d.unit == u) & (d.group_mode == g)].sort_values("delta")
            ls = "-" if u == "ERCB_TUB_H1" else ":"
            ax.plot(t.delta, t.power_fdr05 * 100, ls=ls, marker="o", ms=2.5, color=GCOL[g], lw=0.9)
            rows.append(t)
    ax.axhline(80, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Injected effect (SD)")
    ax.set_ylabel("Genes detected\n(FDR < 0.05, %)")
    ax.set_ylim(-3, 103)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5])
    h = [Line2D([], [], color="black", ls="-", lw=0.9, label="ERCB-TUB-H1"),
         Line2D([], [], color="black", ls=":", lw=0.9, label="ERCB-GLOM-H1")]
    ax.legend(handles=h, loc="upper left", fontsize=5)
    fl.save_source(NAME, "b", pd.concat(rows))


def panel_c(ax, s):
    d = s[(s.gene_mode == "random_noncore") & (s.effect_mode == "constant")]
    rows = []
    for u, lab in UNITS:
        for g, _ in GROUPS:
            t = d[(d.unit == u) & (d.group_mode == g)].sort_values("delta")
            ls = "-" if u == "ERCB_TUB_H1" else ":"
            ax.plot(t.delta, t.auroc_adj, ls=ls, marker="o", ms=2.5, color=GCOL[g], lw=0.9)
            rows.append(t)
    ax.set_xlabel("Injected effect (SD)")
    ax.set_ylabel("Adjusted AUROC of\n50-gene signature")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5])
    fl.save_source(NAME, "c", pd.concat(rows))


def key(ax):
    fl.hide(ax)
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=GCOL[g], label=lab) for g, lab in GROUPS]
    h += [Line2D([], [], marker=mk, ls="", ms=3.5, color="#555555", label=lab) for _, mk, lab in GENES]
    ax.legend(handles=h, loc="center", fontsize=5.5, ncol=4, frameon=False, columnspacing=1.0)


def build():
    s = fl.read("results/21_robustness/A4_injection/summary.tsv")
    fig = fl.new_fig(fl.FULL_W, 72 * fl.MM)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.14], width_ratios=[1.3, 1, 1])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    k = fig.add_subplot(gs[1, :])
    for ax in (a, b, c):
        fl.reserve_label(ax)
    panel_a(a, s)
    panel_b(b, s)
    panel_c(c, s)
    key(k)
    return fl.finalize(fig, {"a": a, "b": b, "c": c})


if __name__ == "__main__":
    fl.run(build, NAME)
