"""Fig. S2 - duplicate deposition and the deduplicated resource (panels a-e).

Inputs (read only): manuscript/numbers/series_table.tsv, results/02_overlap/{duplicate_summary,pair_evidence,
id_expr_agreement,sample_table}.tsv, results/03_labels/sample_labels.tsv,
manuscript/numbers/unique_specimens_by_diagnosis.tsv.
Panel e counts unique specimens (duplicate groups of sample_table) by diagnosis and compartment; the per-diagnosis
totals are asserted to equal manuscript/numbers/unique_specimens_by_diagnosis.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "FigS2"
CONS_COLORS = {"ERCB": "#4C72B0", "NEPTUNE": "#DD8452", "single-centre": "#55A868"}
CALL_COLORS = {"Duplicate (expression)": "#C44E52", "Identifier match only": "#DD8452",
               "Distinct specimens": "#BBBBBB"}
DX_ORDER = ["CONTROL", "DKD", "FSGS", "MN", "IgAN", "MCD", "RPGN", "OTHER", "LN", "HT", "FSGS_MCD", "TMD"]


def panel_a(ax):
    st = fl.read("manuscript/numbers/series_table.tsv")
    st = st.sort_values(["consortium", "series"]).reset_index(drop=True)
    x = np.arange(len(st))
    col = [CONS_COLORS[c] for c in st.consortium]
    ax.bar(x, st.n_samples, color=col, alpha=0.35, width=0.8, edgecolor="white", lw=0.3)
    ax.bar(x, st.n_unique, color=col, width=0.45, edgecolor="white", lw=0.3)
    ax.set_xticks(x, st.series, rotation=90, fontsize=5)
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylabel("Samples")
    ax.set_ylim(0, st.n_samples.max() * 1.25)
    h = [Patch(color=c, label=k) for k, c in CONS_COLORS.items()]
    h += [Patch(color="#999999", alpha=0.35, label="Samples"), Patch(color="#999999", label="Unique specimens within series")]
    ax.legend(handles=h, ncol=5, loc="upper center", fontsize=5.5)
    fl.save_source(NAME, "a", st)
    return st


def panel_b(ax, fig):
    ds = fl.read("results/02_overlap/duplicate_summary.tsv")
    ser = sorted(set(ds.series_a) | set(ds.series_b))
    m = pd.DataFrame(0, index=ser, columns=ser, dtype=float)
    for _, r in ds.iterrows():
        m.loc[r.series_a, r.series_b] = r.n_pairs
        m.loc[r.series_b, r.series_a] = r.n_pairs
    mm = m.where(m > 0)
    im = ax.imshow(mm.values, cmap="Blues", vmin=0, vmax=float(m.values.max()), aspect="equal")
    for i in range(len(ser)):
        for j in range(len(ser)):
            v = m.values[i, j]
            if v > 0:
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=4.2,
                        color="white" if v > 0.55 * m.values.max() else "black")
    ax.set_xticks(range(len(ser)), ser, rotation=90, fontsize=4.8)
    ax.set_yticks(range(len(ser)), ser, fontsize=4.8)
    ax.spines[["top", "right"]].set_visible(True)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=18)
    cb.set_label("Shared specimens", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fl.save_source(NAME, "b", ds.assign(max_shared=int(ds.n_pairs.max())))


def call(r) -> str:
    if r.expr_dup:
        return "Duplicate (expression)"
    if r.same_id is True or r.same_id == "True":
        return "Identifier match only"
    return "Distinct specimens"


def panel_c(ax):
    pe = fl.read("results/02_overlap/pair_evidence.tsv", usecols=["series_a", "gsm_a", "series_b", "gsm_b",
                                                                   "spearman", "resid_r", "resid_rank_in_row",
                                                                   "expr_dup", "same_id", "id_status"])
    pe = pe[pe.resid_rank_in_row == 0].copy()
    pe["call"] = pe.apply(call, axis=1)
    for k in ["Distinct specimens", "Identifier match only", "Duplicate (expression)"]:
        s = pe[pe.call == k]
        ax.scatter(s.spearman, s.resid_r, s=1.5 if k == "Distinct specimens" else 3, lw=0, color=CALL_COLORS[k],
                   rasterized=True, label=f"{k} ({len(s):,})")
    ax.axhline(0.5, ls="--", color="black", lw=0.6)
    ax.set_xlabel("Raw rank correlation")
    ax.set_ylabel("Residual correlation")
    ax.set_ylim(-0.1, 1.25)
    ax.legend(loc="upper left", fontsize=5.5, markerscale=2.5)
    fl.save_source(NAME, "c", pe)


def panel_d(ax):
    ag = fl.read("results/02_overlap/id_expr_agreement.tsv")
    ag = ag.dropna(subset=["n_agree"]).reset_index(drop=True)
    short = {"ID-matched pairs confirmed by expression": "ID pairs confirmed",
             "ID-matched pairs contradicted by expression (different partner)": "ID pairs contradicted",
             "ID-matched pairs without expression call (unconfirmed)": "ID pairs without call",
             "expression-called pairs carrying identical IDs": "Expression pairs, same ID"}
    ag = ag.assign(label=ag.metric.map(short))
    y = np.arange(len(ag))[::-1]
    cols = ["#55A868", "#C44E52", "#BBBBBB", "#4C72B0"]
    ax.barh(y, ag.rate * 100, color=cols[:len(ag)], height=0.6, edgecolor="white", lw=0.3)
    for yi, r in zip(y, ag.itertuples()):
        ax.text(r.rate * 100 + 2, yi, f"{int(r.n_agree)}/{int(r.n)} ({r.rate * 100:.1f}%)", va="center",
                fontsize=5.5)
    ax.set_yticks(y, ag.label, fontsize=6)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pairs (%)")
    fl.save_source(NAME, "d", ag)


def panel_e(ax):
    st = fl.read("results/02_overlap/sample_table.tsv", usecols=["sample_uid", "compartment", "dup_group"])
    lab = fl.read("results/03_labels/sample_labels.tsv", usecols=["sample_uid", "diagnosis"])
    ref = fl.read("manuscript/numbers/unique_specimens_by_diagnosis.tsv").set_index("diagnosis")
    m = st.merge(lab, on="sample_uid", how="left")
    g = m.groupby("dup_group").agg(compartment=("compartment", "first"),
                                   diagnosis=("diagnosis", lambda x: x.dropna().mode().iloc[0]))
    ct = pd.crosstab(g.diagnosis, g.compartment).reindex(DX_ORDER).fillna(0).astype(int)
    assert (ct.sum(1) == ref.loc[ct.index, "n_unique_specimens"]).all(), "specimen counts differ"
    x = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for comp in ["GLOM", "TUB", "WHOLE"]:
        ax.bar(x, ct[comp], bottom=bottom, color=fl.COMP_COLORS[comp], width=0.7, edgecolor="white", lw=0.3,
               label=fl.COMP_LABEL[comp])
        bottom += ct[comp].values
    for xi, t in zip(x, bottom):
        ax.text(xi, t + 4, f"{int(t)}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x, ["Other" if d == "OTHER" else fl.dx_label(d) for d in ct.index], rotation=90)
    ax.set_ylim(0, bottom.max() * 1.55)
    ax.set_xlim(-0.6, len(ct) - 0.4)
    ax.set_ylabel("Unique specimens")
    ax.legend(loc="upper right", fontsize=5.5, ncol=3, title=f"Unique specimens, total {int(bottom.sum()):,}",
              title_fontsize=5.5)
    fl.save_source(NAME, "e", ct.reset_index().assign(total=ct.sum(1).values))


def build():
    fig = fl.new_fig(fl.FULL_W, 162 * fl.MM)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.35, 0.95])
    a = fig.add_subplot(gs[0])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1.25, 1])
    b, c = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    r3 = gs[2].subgridspec(1, 2, width_ratios=[1, 1])
    d, e = fig.add_subplot(r3[0]), fig.add_subplot(r3[1])
    for ax in (a, b, c, d, e):
        fl.reserve_label(ax)
    panel_a(a)
    panel_b(b, fig)
    panel_c(c)
    panel_d(d)
    panel_e(e)
    return fl.finalize(fig, {"a": a, "b": b, "c": c, "d": d, "e": e})


if __name__ == "__main__":
    fl.run(build, NAME)
