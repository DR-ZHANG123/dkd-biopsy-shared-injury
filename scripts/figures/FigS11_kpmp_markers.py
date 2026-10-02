"""Fig. S11 - KPMP-marker validation of response-adjusted cell-type enrichment patterns (panels a-b).

a  cross-cohort (meta) enrichment z of each enrichment pattern with the discovery markers and with KPMP scRNA and
   snRNA markers; glomerular endothelium split into glomerular-capillary (ENDO_GC) and arteriolar (ENDO_ART)
b  per-cohort z with KPMP markers (scRNA | snRNA)
Filled symbols: pattern supported (replication rule). KPMP value = unsplit marker set where it exists,
split-endothelium set for ENDO_GC / ENDO_ART.
Input (read only): results/14_kpmp/claims.tsv
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
CT = {"PODO": "Podocyte", "ENDO": "Endothelial", "ENDO_PT": "Peritub. endo.", "MAST": "Mast",
      "ENDO_GC": "Glom.-capillary endo.", "ENDO_ART": "Arteriolar endo.", "MAC": "Macrophage", "TCELL": "T cell"}
REFS = [("old", "Discovery atlases", "#8C8C8C", "o"), ("scRNA", "KPMP scRNA", "#DD8452", "s"),
        ("snRNA", "KPMP snRNA", "#4C72B0", "^")]


def tidy() -> pd.DataFrame:
    c = fl.read("results/14_kpmp/claims.tsv")
    rows = []
    for _, r in c.iterrows():
        for ref, *_ in REFS:
            if ref == "old":
                z, sup = r.old_z, r.old_supported
            else:
                z, sup = r[f"z_{ref}"], r[f"supported_{ref}"]
                if pd.isna(z):
                    z, sup = r[f"z_{ref}_endosplit"], r[f"supported_{ref}_endosplit"]
            rows.append(dict(compartment=r.compartment, contrast=r.contrast, cell_type=r.cell_type,
                             expected=r.expected, unit=r.unit, reference=ref, z=z,
                             supported=bool(sup) if not pd.isna(sup) else np.nan))
    d = pd.DataFrame(rows)
    d["claim"] = [f"{x.split('_')[0]} {'GLOM' if c == 'GLOM' else 'TUB'}: {CT.get(t, t)}"
                  for x, c, t in zip(d.contrast, d.compartment, d.cell_type)]
    return d


def claim_order(d):
    return list(dict.fromkeys(d.claim))


def panel_a(ax, d):
    m = d[d.unit == "META"]
    order = claim_order(m)[::-1]
    ypos = {c: i for i, c in enumerate(order)}
    for k, (ref, lab, col, mk) in enumerate(REFS):
        s = m[(m.reference == ref) & m.z.notna()]
        y = s.claim.map(ypos) + (k - 1) * 0.22
        filled = s.supported.map(lambda v: v is True).astype(bool)
        ax.scatter(s.z[filled], y[filled], s=12, marker=mk, color=col, edgecolor="none", zorder=3)
        ax.scatter(s.z[~filled], y[~filled], s=12, marker=mk, facecolor="white", edgecolor=col, lw=0.7, zorder=3)
    for lim in (-2, 2):
        ax.axvline(lim, ls=":", color="#888888", lw=0.5)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)), order, fontsize=5.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Cross-cohort enrichment z")
    ax.set_xlim(-11, 15)
    hs = [Line2D([], [], marker=mk, ls="", ms=3.5, color=col, label=lab) for _, lab, col, mk in REFS]
    hs.append(Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", mec="#555555", label="not supported"))
    ax.legend(handles=hs, loc="lower right", fontsize=5.5)
    fl.save_source(NAME, "a", m.assign(y=m.claim.map(ypos)))


def panel_b(ax, d):
    u = d[(d.unit != "META") & d.reference.isin(["scRNA", "snRNA"])]
    order = claim_order(d[d.unit == "META"])
    cols = [("scRNA", "H1"), ("scRNA", "H7"), ("snRNA", "H1"), ("snRNA", "H7")]
    mat = np.full((len(order), len(cols)), np.nan)
    for i, cl in enumerate(order):
        for j, (ref, suf) in enumerate(cols):
            v = u[(u.claim == cl) & (u.reference == ref) & u.unit.str.endswith(suf)]
            if len(v) and not pd.isna(v.z.iloc[0]):
                mat[i, j] = v.z.iloc[0]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    for i in range(len(order)):
        for j in range(len(cols)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.5, color="#888888")
            else:
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(mat[i, j]) > 6 else "black")
    ax.axvline(1.5, color="white", lw=2)
    ax.set_xticks(range(len(cols)), [f"{r}\n{s}" for r, s in cols])
    ax.set_yticks(range(len(order)), order, fontsize=5.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.45, pad=0.03, aspect=14)
    cb.set_label("z, ERCB cohort (KPMP markers)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fl.save_source(NAME, "b", u[["claim", "unit", "reference", "z", "supported"]])


def build():
    d = tidy()
    fig = fl.new_fig(fl.FULL_W, 95 * fl.MM)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fl.reserve_label(a)
    fl.reserve_label(b)
    panel_a(a, d)
    panel_b(b, d)
    return fl.finalize(fig, {"a": a, "b": b})


if __name__ == "__main__":
    fl.run(build, NAME)
