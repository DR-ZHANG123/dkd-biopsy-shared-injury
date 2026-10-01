"""Fig. 6 - small disease-specific residuals after accounting for the shared program (panels a-b).

a  program-adjusted cell-type enrichment patterns (diagnosis vs other patients); b  residual signatures in independent
cohorts. Published-signature panels are in Fig. S12 (FigS12_signatures.py).
Inputs (read only): manuscript/numbers/celltype_meta_z.tsv, manuscript/numbers/celltype_replication.tsv,
results/14_kpmp/claims.tsv, results/19_shared_program/figdata/fig4_residual_tests.tsv.
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

NAME = "Fig6"
CT_ORDER = ["PODO", "PEC", "ENDO", "ENDO_PT", "PERI", "STROMA", "PT", "LOH", "DCT", "CNT_PC", "IC",
            "MAC", "TCELL", "NK", "NEUTRO", "MAST"]
CT_LABEL = {"PODO": "Podocyte", "PEC": "PEC", "ENDO": "Endothelial", "ENDO_PT": "Peritubular EC",
            "PERI": "Pericyte", "STROMA": "Stromal", "PT": "PT", "LOH": "LOH", "DCT": "DCT", "CNT_PC": "CNT/PC",
            "IC": "IC", "MAC": "Macrophage", "TCELL": "T cell", "NK": "NK", "NEUTRO": "Neutrophil",
            "MAST": "Mast cell"}
DX_ROWS = ["DKD", "FSGS", "FSGS_MCD", "MCD", "MN", "IgAN", "LN", "HT", "RPGN"]
STATUS_COL = {"replicated": "#55A868", "not_replicated": "#C44E52", "descriptive": "#8C8C8C"}


def panel_a(axs, cax):
    z = fl.read("manuscript/numbers/celltype_meta_z.tsv")
    rep = fl.read("manuscript/numbers/celltype_replication.tsv")
    cl = fl.read("results/14_kpmp/claims.tsv")
    cl = cl[cl.unit == "META"].copy()
    conf = []
    for _, q in cl.iterrows():
        refs = [str(q[f"supported_{r}"]) == "True" for r in ("scRNA", "snRNA") if pd.notna(q[f"z_{r}"])]
        conf.append(bool(refs) and all(refs))
    cl["kpmp_confirmed"] = conf  # supported in every KPMP reference (snRNA, scRNA) that could test it
    cl["diagnosis"] = cl.contrast.str.replace("_vs_PAT_adj", "", regex=False)
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
    out = []
    for ax, comp in zip(axs, ["GLOM", "TUB"]):
        d = z[(z.compartment == comp) & z.contrast.str.endswith("_vs_PAT_adj")].copy()
        d["diagnosis"] = d.contrast.str.replace("_vs_PAT_adj", "", regex=False)
        rows = [x for x in DX_ROWS if x in set(d.diagnosis)]
        m = d.set_index("diagnosis").loc[rows, CT_ORDER].astype(float)
        im = ax.imshow(m.values.clip(-8, 8), cmap="RdBu_r", norm=norm, aspect="auto")
        nanm = np.ma.masked_where(~m.isna().values, np.ones(m.shape))
        ax.imshow(nanm, cmap="Greys", vmin=0, vmax=4, aspect="auto")
        r = rep[(rep.compartment == comp) & rep.replicated]
        k = cl[(cl.compartment == comp) & cl.kpmp_confirmed]
        for i, dx in enumerate(rows):
            for j, ct in enumerate(CT_ORDER):
                isr = ((r.diagnosis == dx) & (r.cell_type == ct)).any()
                isk = ((k.diagnosis == dx) & (k.cell_type == ct)).any()
                if isr:
                    ax.plot(j, i, marker="o", ms=2.2, color="black", mec="none")
                if isk:
                    ax.plot(j, i, marker="s", ms=6.2, mfc="none", mec="black", mew=0.7)
                out.append(dict(compartment=comp, diagnosis=dx, cell_type=ct, z_adj=m.loc[dx, ct],
                                replicated=isr, kpmp_confirmed=isk))
        ax.set_xticks(range(len(CT_ORDER)), [CT_LABEL[c] for c in CT_ORDER], rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), [fl.dx_label(x) for x in rows], fontsize=6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{fl.COMP_LABEL[comp]} (adjusted, vs other patients)", fontsize=5.8, pad=7)
    cb = axs[0].figure.colorbar(im, cax=cax)
    cb.set_label("Marker enrichment z", fontsize=6, fontweight="bold")
    cb.ax.tick_params(labelsize=5.5, length=2)
    h = [Line2D([], [], marker="o", ls="", ms=2.2, color="black", label="Replicated (≥2 cohorts)"),
         Line2D([], [], marker="s", ls="", ms=5, mfc="none", mec="black", label="Confirmed with KPMP markers")]
    leg = axs[1].legend(handles=h, loc="upper left", bbox_to_anchor=(1.0, -0.01), fontsize=5.5, frameon=False)
    leg.set_gid("free")
    fl.save_source(NAME, "a", pd.DataFrame(out))


def panel_b(ax):
    t = fl.read("results/19_shared_program/figdata/fig4_residual_tests.tsv")
    t = t.assign(cohort_l=t.cohort.map(fl.unit_label).str.replace("KPMP", "KPMP sections"))
    order = {"replicated": 0, "not_replicated": 1, "descriptive": 2}
    t = t.assign(o=t.status.map(order)).sort_values(["o", "disease", "cohort_l"], ascending=[False, False, False])
    t = t.reset_index(drop=True)
    y = np.arange(len(t))
    ax.hlines(y, t.null_median, t.null_q95, color="#CCCCCC", lw=2.5, zorder=1)
    for st, c in STATUS_COL.items():
        s = t[t.status == st]
        ax.scatter(s.auroc, s.index, s=14, color=c if st != "descriptive" else "white", edgecolor=c,
                   lw=0.8, zorder=3, label=st.replace("_", " ").capitalize())
    ax.axvline(0.5, ls="--", color="black", lw=0.5)
    ax.set_yticks(y, [f"{dx} · {c} (n={p})" for dx, c, p in zip(t.disease, t.cohort_l, t.n_pos)], fontsize=5.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xlabel("Within-patient AUROC")
    ax.plot([], [], color="#CCCCCC", lw=2.5, label="Random signatures")
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.44), fontsize=5.3, frameon=False, handlelength=1.0)
    fl.save_source(NAME, "b", t.drop(columns="o"))


def build():
    fig = fl.new_fig(fl.FULL_W, 140 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85])
    r1 = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 0.035])
    a1, a2, cax = fig.add_subplot(r1[0]), fig.add_subplot(r1[1]), fig.add_subplot(r1[2])
    b = fig.add_subplot(gs[1])
    fl.reserve_label(b)
    panel_a([a1, a2], cax)
    panel_b(b)
    return fl.finalize(fig, {"a": a1, "b": b})


if __name__ == "__main__":
    fl.run(build, NAME)
