"""Fig. S8 - failed-repair programs in bulk biopsy cohorts (panels a-c).

Inputs (read only): results/20_repair_state/figdata/panel_f_bulk_meta.tsv, panel_f_bulk_scp_relation.tsv,
panel_f_bulk_resolvability.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS8"
FD = "results/20_repair_state/figdata/"
DISEASES = ["DKD", "RPGN", "FSGS", "IgAN", "LN", "MN", "MCD", "HT", "nonDKD_patients"]
DLAB = {"nonDKD_patients": "All non-DKD"}
MEAS = [("scp", "Injury–repair score", "#8C8C8C", "D"), ("prog_noSCP_PT:rfPT", "aPT+frPT program (no response genes)", "#C44E52", "o"),
        ("prog_noSCP_TAL:rfTAL", "aTAL+frTAL program (no response genes)", "#DD8452", "s")]
UNITS = ["ERCB_TUB_H1", "ERCB_TUB_H7", "GSE108112-GPL19983", "GSE30529-GPL571CEL", "GSE133288-GPL19983",
         "GSE182380-RNAseq", "GSE175759-RNAseq", "GSE142025-RNAseq", "GSE115857-GPL14951"]


def panel_a(ax, m):
    rows = []
    for yi, dz in enumerate(DISEASES):
        for k, (meas, _, col, mk) in enumerate(MEAS):
            r = m[(m.disease == dz) & (m.measure == meas)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = yi + (k - 1) * 0.26
            ax.plot([r.re_lo, r.re_hi], [y, y], color=col, lw=0.8)
            ax.plot([r.re_mu], [y], ls="", marker=mk, ms=3, color=col)
            rows.append(r)
        n = m[(m.disease == dz) & (m.measure == "scp")].n_units.iloc[0]
        ax.text(6.25, yi, f"k = {n:.0f}", ha="right", va="center", fontsize=5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(DISEASES)), [DLAB.get(d, d) for d in DISEASES], fontsize=5.5)
    ax.set_ylim(len(DISEASES) - 0.5, -1.9)
    ax.set_xlim(-0.8, 6.3)
    ax.set_xlabel("Disease vs control Hedges g (random effects)")
    h = [Line2D([], [], ls="-", marker=mk, ms=3, lw=0.8, color=col, label=lab) for _, lab, col, mk in MEAS]
    ax.legend(handles=h, loc="upper right", fontsize=5, handletextpad=0.3, borderpad=0.25)
    fl.save_source(NAME, "a", pd.DataFrame(rows))


def panel_b(ax, s):
    s = s[(s.samples == "patients") & s.measure.isin([MEAS[1][0], MEAS[2][0]])]
    rows = []
    for k, (meas, lab, col, mk) in enumerate(MEAS[1:]):
        t = s[s.measure == meas].set_index("unit").reindex(UNITS)
        y = np.arange(len(UNITS)) + (k - 0.5) * 0.3
        ax.plot(t.spearman_vs_scp, y, ls="", marker=mk, ms=3, color=col, label=lab)
        rows.append(t.reset_index())
    ax.set_yticks(range(len(UNITS)), [fl.unit_label(u) for u in UNITS], fontsize=5.5)
    ax.set_ylim(len(UNITS) - 0.5, -0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Spearman ρ with injury–repair\nscore (patients)")
    fl.save_source(NAME, "b", pd.concat(rows))


def panel_c(ax, rv):
    rows = []
    for k, (state, col, mk) in enumerate((("PT:rfPT", "#C44E52", "o"), ("TAL:rfTAL", "#DD8452", "s"))):
        t = rv[rv.state == state].set_index("unit").reindex(UNITS)
        y = np.arange(len(UNITS)) + (k - 0.5) * 0.3
        ax.plot(t.median_share * 100, y, ls="", marker=mk, ms=3, color=col)
        rows.append(t.reset_index())
    ax.axvline(1, color="#999999", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_yticks(range(len(UNITS)), [])
    ax.set_ylim(len(UNITS) - 0.5, -0.6)
    ax.set_xlabel("Median BayesPrism\nshare (% of lineage)")
    fl.save_source(NAME, "c", pd.concat(rows))


def build():
    m = fl.read(FD + "panel_f_bulk_meta.tsv")
    s = fl.read(FD + "panel_f_bulk_scp_relation.tsv")
    rv = fl.read(FD + "panel_f_bulk_resolvability.tsv")
    fig = fl.new_fig(fl.FULL_W, 80 * fl.MM)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.75])
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    for ax in (a, b, c):
        fl.reserve_label(ax)
    panel_a(a, m)
    panel_b(b, s)
    panel_c(c, rv)
    return fl.finalize(fig, {"a": a, "b": b, "c": c})


if __name__ == "__main__":
    fl.run(build, NAME)
