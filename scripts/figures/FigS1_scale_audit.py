"""Fig. S1 - scale audit of the expression matrices (panels a, b).

Input (read only): results/01_ingest/series_scale_audit.tsv.
The table holds the statistics of the matrix that entered the analysis (after replacement or CEL
reprocessing) together with the classification of the deposited matrix (`deposited_scale`) and the action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "FigS1"
ACT_COLORS = {"No change": "#8C8C8C", "Replaced by non-normalized table": "#DD8452",
              "Excluded": "#C44E52", "Reprocessed from CEL (RMA)": "#4C72B0", "RNA-seq log2(CPM+1)": "#55A868"}


def action_class(a: str) -> str:
    if a.startswith("excluded"):
        return "Excluded"
    if "replaced by GEO non-normalized" in a:
        return "Replaced by non-normalized table"
    if "CEL RMA" in a:
        return "Reprocessed from CEL (RMA)"
    if "CPM" in a:
        return "RNA-seq log2(CPM+1)"
    return "No change"


def load():
    d = fl.read("results/01_ingest/series_scale_audit.tsv")
    d = d.assign(action_class=d.action.map(action_class), ratio=d.sd_row_mean / d.median_col_sd,
                 deposited=[dep if isinstance(dep, str) else ("probe_centred" if "probe-centred" in act else
                            "counts") for dep, act in zip(d.deposited_scale, d.action)])
    return d


def panel_a(ax, d):
    for k, c in ACT_COLORS.items():
        s = d[d.action_class == k]
        ax.scatter(s.median_col_sd, s.sd_row_mean, s=14, color=c, edgecolor="white", lw=0.3, zorder=3)
    x = np.logspace(0, 3.6, 50)
    ax.plot(x, 0.25 * x, ls="--", color="black", lw=0.6)
    ax.plot(x, x, ls=":", color="#999999", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 4000)
    ax.set_ylim(0.2, 4000)
    ax.text(3500, 12, "flag threshold\n(0.25 \u00d7 within-sample SD)", ha="right", va="top", fontsize=5.5)
    ax.text(20, 40, "y = x", ha="right", va="bottom", fontsize=5.5, color="#777777", rotation=0)
    ax.set_xlabel("Median within-sample SD")
    ax.set_ylabel("SD of gene means")
    for _, r in d[d.action_class.isin(["Excluded", "Replaced by non-normalized table"])].iterrows():
        dy = 2.2 if r.action_class == "Excluded" else 0.45
        ax.annotate(r.series.split("-")[0], (r.median_col_sd, r.sd_row_mean), xytext=(r.median_col_sd / 12,
                    r.sd_row_mean * dy), fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"),
                    ha="right", va="center")
    fl.save_source(NAME, "a", d[["series", "technology", "gpl", "sd_row_mean", "median_col_sd", "ratio", "scale",
                                 "deposited", "action_class", "action"]])


def panel_b(ax, d):
    d = d.sort_values(["action_class", "series"]).reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.scatter(d.ratio, y, s=12, color=[ACT_COLORS[a] for a in d.action_class], edgecolor="white", lw=0.3,
               zorder=3)
    ax.axvline(0.25, ls="--", color="black", lw=0.6)
    lab = [f"{s}  [deposited: {dep}]" for s, dep in zip(d.series, d.deposited)]
    ax.set_yticks(y, lab, fontsize=5)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("SD of gene means / within-sample SD")
    handles = [Line2D([], [], marker="o", ls="", ms=3.5, color=c, label=k) for k, c in ACT_COLORS.items()]
    leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.12), fontsize=5.5,
                    ncol=3, frameon=False, columnspacing=1.0)
    leg.set_gid("free")
    fl.save_source(NAME, "b", d[["series", "deposited", "ratio", "action_class", "action"]])
    return leg


def build():
    d = load()
    fig = fl.new_fig(fl.FULL_W, 95 * fl.MM)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (a, b):
        fl.reserve_label(ax)
    panel_a(a, d)
    panel_b(b, d)
    return fl.finalize(fig, {"a": a, "b": b})


if __name__ == "__main__":
    fl.run(build, NAME)
