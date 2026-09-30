"""Fig. 3 - composition and cell state in the shared program (panels a-e).

Inputs (read only): results/19_shared_program/figdata/fig2_composition_r2.tsv,
fig2_kpmp_decomposition.tsv, fig2_core_localisation.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "Fig3"
FD = "results/19_shared_program/figdata/"
REF_COL = {"scRNA": "#4C72B0", "snRNA": "#DD8452"}
REF_LAB = {"scRNA": "scRNA reference", "snRNA": "snRNA reference"}
CT_LABEL = {"PODO": "Podocyte", "PEC": "PEC", "MES": "Mesangial", "ENDO": "Endothelial",
            "ENDO_PT": "Peritubular EC", "PT": "PT", "PT_injured": "Injured PT", "LOH": "LOH", "DCT": "DCT",
            "CNT_PC": "CNT/PC", "IC": "IC", "STROMA": "Stromal", "PERI": "Pericyte", "MAC": "Macrophage",
            "DC": "DC", "TCELL": "T cell", "NK": "NK", "BCELL": "B cell", "MAST": "Mast", "NEUTRO": "Neutrophil"}
CT_ORDER = ["PODO", "PEC", "MES", "ENDO", "ENDO_PT", "PT", "PT_injured", "LOH", "DCT", "CNT_PC", "IC",
            "STROMA", "PERI", "MAC", "DC", "TCELL", "NK", "BCELL", "MAST"]
KIND_COL = {"full": "#8C8C8C", "comp_only": "#55A868", "state_only": "#8172B3"}
KIND_LAB = {"full": "Full", "comp_only": "Compositional only", "state_only": "Within-lineage only"}


def unit_order(d):
    units = []
    for comp in ("GLOM", "TUB"):
        units += list(dict.fromkeys(d[d.compartment == comp].unit))
    return units


def panel_a(ax, d):
    units = unit_order(d)
    xs = {u: i for i, u in enumerate(units)}
    rows = []
    for k, (ref, off) in enumerate((("scRNA", -0.17), ("snRNA", 0.17))):
        s = d[d.reference == ref].assign(x=lambda t: t.unit.map(xs) + off)
        ax.scatter(s.x, s.r2_cv, s=12, color=REF_COL[ref], zorder=3, lw=0)
        ax.hlines(s.r2_cv_random_q95, s.x - 0.14, s.x + 0.14, color=REF_COL[ref], lw=1.0, alpha=0.6, zorder=2)
        rows.append(s[["reference", "compartment", "unit", "n", "r2_cv", "r2_cv_random_median",
                       "r2_cv_random_q95", "x"]])
    nglom = int((d[d.reference == "scRNA"].compartment == "GLOM").sum())
    ax.axvline(nglom - 0.5, color="#BBBBBB", lw=0.5)
    ax.text(nglom / 2 - 0.5, 1.13, "Glomerulus", ha="center", va="center", fontsize=6.5)
    ax.text(nglom + (len(units) - nglom) / 2 - 0.5, 1.13, "Tubulointerstitium", ha="center", va="center",
            fontsize=6.5)
    ax.axhline(0, color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_xticks(range(len(units)), [fl.unit_label(u) for u in units], rotation=90, fontsize=5.5)
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.set_ylim(-1.05, 1.2)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_ylabel("Cross-validated R$^2$")
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=REF_COL[r], label=REF_LAB[r]) for r in REF_COL]
    h.append(Line2D([], [], color="#8C8C8C", lw=1.0, label="Random gene sets, 95th pct"))
    ax.legend(handles=h, loc="lower left", ncol=3, fontsize=5.5)
    med = d.groupby(["reference", "compartment"]).r2_cv.median()
    fl.save_source(NAME, "a", pd.concat(rows).assign(
        median_ref_comp=lambda t: [med[(r, c)] for r, c in zip(t.reference, t.compartment)]))


def panel_b(ax, d):
    d = d[d.auroc_raw.notna()]
    xpos = {"raw": 0, "scRNA": 1, "snRNA": 2}
    rows = []
    for (comp, unit), g in d.groupby(["compartment", "unit"]):
        raw = g.auroc_raw.iloc[0]
        for ref in ("scRNA", "snRNA"):
            adj = g[g.reference == ref].auroc_composition_adjusted.iloc[0]
            ax.plot([0, xpos[ref]], [raw, adj], color=fl.COMP_COLORS[comp], lw=0.4, alpha=0.5, zorder=1)
            rows.append(dict(compartment=comp, unit=unit, reference=ref, auroc_raw=raw,
                             auroc_composition_adjusted=adj))
    rows = pd.DataFrame(rows)
    for comp, off in (("GLOM", -0.08), ("TUB", 0.08)):
        r = rows[rows.compartment == comp]
        raw = r.drop_duplicates("unit").auroc_raw
        ax.scatter(np.zeros(len(raw)) + off, raw, s=9, color=fl.COMP_COLORS[comp], zorder=3, lw=0)
        ax.hlines(raw.median(), -0.3 + off, 0.3 + off, color=fl.COMP_COLORS[comp], lw=1.0)
        for ref in ("scRNA", "snRNA"):
            v = r[r.reference == ref].auroc_composition_adjusted
            ax.scatter(np.full(len(v), xpos[ref] + off), v, s=9, color=fl.COMP_COLORS[comp], zorder=3, lw=0)
            ax.hlines(v.median(), xpos[ref] - 0.3 + off, xpos[ref] + 0.3 + off, color=fl.COMP_COLORS[comp],
                      lw=1.0)
    ax.axhline(0.5, ls="--", color="black", lw=0.6)
    ax.set_xticks([0, 1, 2], ["Raw", "Adjusted\n(scRNA)", "Adjusted\n(snRNA)"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.35, 1.08)
    ax.set_ylabel("Patient vs control AUROC")
    h = [Line2D([], [], marker="o", ls="", ms=3, color=fl.COMP_COLORS[c], label=fl.COMP_LABEL[c])
         for c in ("GLOM", "TUB")]
    ax.legend(handles=h, loc="lower left", fontsize=5.5)
    med = rows.groupby(["compartment", "reference"])[["auroc_raw", "auroc_composition_adjusted"]].median()
    fl.save_source(NAME, "b", rows.merge(med.add_suffix("_median").reset_index(), on=["compartment", "reference"]))


def panel_c(ax, k):
    k = k[(k.contrast == "CKD_vs_REF") & k.kind.isin(["full", "comp_only", "state_only"])]
    groups = [("snRNA", "GLOM"), ("snRNA", "TUB"), ("scRNA", "GLOM"), ("scRNA", "TUB")]
    rows = []
    for gi, (ds, comp) in enumerate(groups):
        g = k[(k.dataset == ds) & (k.compartment == comp)].set_index("kind")
        full = g.loc["full", "delta"]
        for j, kind in enumerate(("full", "comp_only", "state_only")):
            x = gi + (j - 1) * 0.27
            v = g.loc[kind, "delta"]
            ax.bar(x, v * 100, width=0.26, color=KIND_COL[kind], edgecolor="white", lw=0.3)
            share = v / full
            if kind != "full":
                ax.text(x, max(v * 100, 0) + 0.25, f"{share * 100:.0f}%", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)
            rows.append(dict(dataset=ds, compartment=comp, kind=kind, delta=v, share_of_full=share,
                             n_ckd=g.loc[kind, "n_A"], n_ref=g.loc[kind, "n_B"], p_perm=g.loc[kind, "p_perm"],
                             auroc=g.loc[kind, "auroc"]))
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(4), [f"{ds}\n{c}" for ds, c in groups], fontsize=5.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-1.0, 13.5)
    ax.set_ylabel("CKD − reference injury–repair score (×10$^{-2}$)")
    h = [Patch(color=KIND_COL[x], label=KIND_LAB[x]) for x in KIND_COL]
    ax.legend(handles=h, loc="upper left", fontsize=5.5)
    fl.save_source(NAME, "c", pd.DataFrame(rows))


def panel_d(axs, k):
    k = k[(k.contrast == "CKD_vs_REF") & (k.dataset == "snRNA")]
    rows = []
    y = np.arange(len(CT_ORDER))
    for ax, comp in zip(axs, ("GLOM", "TUB")):
        g = k[k.compartment == comp]
        full = g[g.kind == "full"].delta.iloc[0]
        st = g[g.kind == "state_by_type"].set_index("cell_type").delta.reindex(CT_ORDER) / full * 100
        cp = g[g.kind == "comp_by_type"].set_index("cell_type").delta.reindex(CT_ORDER) / full * 100
        ax.barh(y - 0.2, st, height=0.38, color=KIND_COL["state_only"], edgecolor="white", lw=0.3)
        ax.barh(y + 0.2, cp, height=0.38, color=KIND_COL["comp_only"], edgecolor="white", lw=0.3)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(y, [CT_LABEL[c] for c in CT_ORDER] if comp == "GLOM" else [], fontsize=5.5)
        ax.set_ylim(len(CT_ORDER) - 0.4, -0.6)
        ax.set_xlim(-3, 22)
        ax.set_xlabel("% of full difference")
        ax.set_title(fl.COMP_LABEL[comp], fontsize=6.5, pad=3)
        rows.append(pd.DataFrame(dict(compartment=comp, cell_type=CT_ORDER, state_pct_of_full=st.values,
                                      comp_pct_of_full=cp.values, full_delta=full)))
    fl.save_source(NAME, "d", pd.concat(rows))


def panel_e(ax, lo):
    lo = lo[lo.dataset == "snRNA"]
    cols = [("GLOM", "up"), ("GLOM", "down"), ("TUB", "up"), ("TUB", "down")]
    m = np.full((len(CT_ORDER), 4), np.nan)
    sig = np.zeros_like(m, dtype=bool)
    rows = []
    for j, (comp, dr) in enumerate(cols):
        g = lo[(lo.compartment == comp) & (lo.direction == dr)].set_index("top_type").reindex(CT_ORDER)
        m[:, j] = np.log2(g.odds_ratio.clip(lower=1e-3))
        sig[:, j] = g.fdr < 0.05
        rows.append(g.reset_index().rename(columns={"index": "top_type"}).assign(compartment=comp, direction=dr,
                                                                                 log2_or=m[:, j]))
    lim = 3.0
    im = ax.imshow(np.clip(m, -lim, lim), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(4):
            if sig[i, j] and np.isfinite(m[i, j]):
                ax.text(j, i, f"{2 ** m[i, j]:.2g}", ha="center", va="center", fontsize=4.5,
                        color="white" if abs(m[i, j]) > 1.8 else "black")
    ax.set_xticks(range(4), ["GLOM up", "GLOM down", "TUB up", "TUB down"], rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(CT_ORDER)), [CT_LABEL[c] for c in CT_ORDER], fontsize=5.5)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.45, aspect=12, pad=0.03)
    cb.set_label("log$_2$ odds ratio", fontsize=5.5, fontweight="bold")
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    fl.save_source(NAME, "e", pd.concat(rows))


def build():
    d = fl.read(FD + "fig2_composition_r2.tsv")
    k = fl.read(FD + "fig2_kpmp_decomposition.tsv")
    lo = fl.read(FD + "fig2_core_localisation.tsv")
    fig = fl.new_fig(fl.FULL_W, 150 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35])
    r1 = gs[0].subgridspec(1, 2, width_ratios=[2.3, 1])
    a, b = fig.add_subplot(r1[0]), fig.add_subplot(r1[1])
    r2 = gs[1].subgridspec(1, 4, width_ratios=[1.05, 1.0, 0.8, 1.05])
    c = fig.add_subplot(r2[0])
    d1, d2 = fig.add_subplot(r2[1]), fig.add_subplot(r2[2])
    e = fig.add_subplot(r2[3])
    for ax in (a, b, c, e):
        fl.reserve_label(ax)
    panel_a(a, d)
    panel_b(b, d)
    panel_c(c, k)
    panel_d([d1, d2], k)
    h = [Patch(color=KIND_COL["state_only"], label="Within-lineage"), Patch(color=KIND_COL["comp_only"], label="Compositional")]
    d2.legend(handles=h, loc="lower right", fontsize=5.5)
    panel_e(e, lo)
    return fl.finalize(fig, {"a": a, "b": b, "c": c, "d": d1, "e": e})


if __name__ == "__main__":
    fl.run(build, NAME)
