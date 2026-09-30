"""Fig. 5 - independent replication and clinical association of the shared program (panels a-c).

Inputs (read only): results/19_shared_program/figdata/fig3_replication_auroc.tsv,
fig3_kpmp_egfr_scatter.tsv, fig3_clinical.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

NAME = "Fig5"
FD = "results/19_shared_program/figdata/"
COHORTS = ["GSE175759-RNAseq", "GSE142025-RNAseq", "GSE115857-GPL14951", "GSE166239", "KPMP", "GSE162830"]
COH_LAB = {"KPMP": "KPMP sections"}
GS_COL = {"full": "#4C72B0", "no_procurement": "#DD8452", "no_ieg": "#55A868"}
GS_LAB = {"full": "All response genes", "no_procurement": "Procurement-sensitive removed",
          "no_ieg": "Immediate-early removed"}


def fp(p: float) -> str:
    return "P < 0.001" if p < 0.001 else f"P = {p:.2g}"


def num(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace("-", "\u2212")


def pct(v: float) -> str:
    return f"{v * 100:.2f}".rstrip("0").rstrip(".")


def coh(c: str) -> str:
    return COH_LAB.get(c, fl.unit_label(c))


def panel_a(ax, r):
    r = r[r.primary & ~r.supplementary & r.cohort.isin(COHORTS)]
    rows = []
    for i, c in enumerate(COHORTS):
        g = r[r.cohort == c]
        for j, gs in enumerate(GS_COL):
            s = g[g.gene_set == gs].iloc[0]
            x = i + (j - 1) * 0.25
            ax.add_patch(Rectangle(
                (x - 0.1, s.null_median), 0.2, s.null_q95 - s.null_median, color="#DDDDDD", lw=0, zorder=1))
            ax.errorbar(x, s.auroc, yerr=[[s.auroc - s.auroc_lo], [s.auroc_hi - s.auroc]], fmt="o", ms=3.5,
                        color=GS_COL[gs], elinewidth=0.7, capsize=1.5, capthick=0.6, zorder=3)
            rows.append(s.to_dict() | {"x": x})
        full = g[g.gene_set == "full"].iloc[0]
        ax.text(i, 0.335, f"percentile {pct(full.pct_vs_random)}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.5, ls="--", color="black", lw=0.6)
    labs = [f"{coh(c)}\n{int(r[r.cohort == c].n_pos.iloc[0])} vs {int(r[r.cohort == c].n_ctrl.iloc[0])}"
            for c in COHORTS]
    ax.set_xticks(range(len(COHORTS)), labs, fontsize=6)
    ax.set_xlim(-0.55, len(COHORTS) - 0.45)
    ax.set_ylim(0.3, 1.12)
    ax.set_ylabel("Disease vs healthy AUROC")
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=GS_COL[g], label=GS_LAB[g]) for g in GS_COL]
    h.append(Patch(color="#DDDDDD", label="Random gene sets, median–95th pct"))
    ax.legend(handles=h, loc="upper center", ncol=4, fontsize=5.5, bbox_to_anchor=(0.5, 1.0))
    fl.save_source(NAME, "a", pd.DataFrame(rows))


def panel_b(ax, s, cl):
    s = s[(s.compartment == "TUB") & (s.scenario == "full") & s.group.isin(["REF", "OTHER_CKD", "DKD"])]
    s = s[s.egfr_mid.notna()]
    sty = {"REF": ("#BBBBBB", "Reference"), "OTHER_CKD": ("#4C72B0", "Other CKD"), "DKD": ("#C44E52", "DKD")}
    for g, (col, lab) in sty.items():
        d = s[s.group == g]
        ax.scatter(d.egfr_mid + fl.jitter(len(d), 1.5), d.score, s=8, color=col, lw=0, alpha=0.9,
                   label=f"{lab} (n = {len(d)})", zorder=3 if g == "DKD" else 2)
    c = cl[(cl.cohort == "KPMP_snRNA_donors") & (cl.core == "TUB") & (cl.gene_set == "full")].set_index("test")
    ckd, dkd = c.loc["egfr_bin_CKD"], c.loc["egfr_bin_DKD"]
    txt = (f"CKD ρ = {num(ckd.spearman)}, {fp(ckd.p)}, n = {int(ckd.n)}\n"
           f"DKD ρ = {num(dkd.spearman)}, {fp(dkd.p)}, n = {int(dkd.n)}")
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
    ax.set_xlabel("eGFR (mL/min/1.73 m$^2$)")
    ax.set_ylabel("Injury–repair score (TUB genes)")
    ylo, yhi = s.score.min(), s.score.max()
    ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.35 * (yhi - ylo))
    ax.set_xlim(0, s.egfr_mid.max() + 45)
    ax.legend(loc="upper right", fontsize=5.3, bbox_to_anchor=(1.0, 0.86))
    fl.save_source(NAME, "b", s.assign(rho_CKD=ckd.spearman, p_CKD=ckd.p, n_CKD=ckd.n, rho_DKD=dkd.spearman,
                                       p_DKD=dkd.p, n_DKD=dkd.n))


TESTS = [("KPMP", "proteinuria_cat_patients", "Proteinuria, KPMP sections", "rho"),
         ("GSE166239", "proteinuria_patients", "Proteinuria, GSE166239", "rho"),
         ("GSE142025-RNAseq", "advanced_vs_early_DN", "Advanced vs early DKD, GSE142025", "auc"),
         ("GSE115857-GPL14951", "iga_grade_G1_G3", "IgAN grade, GSE115857", "rho"),
         ("KPMP_snRNA_donors", "egfr_bin_CKD", "eGFR, KPMP snRNA CKD donors", "rho"),
         ("KPMP_snRNA_donors", "egfr_bin_DKD", "eGFR, KPMP snRNA DKD donors", "rho"),
         ("GSE175759-RNAseq", "egfr_patients", "eGFR, GSE175759", "rho"),
         ("KPMP", "egfr_bin_patients", "eGFR, KPMP sections", "rho"),
         ("GSE166239", "egfr_patients", "eGFR, GSE166239", "rho")]


def panel_c(ax, cl):
    cl = cl[cl.primary]
    rows = []
    for i, (c, t, lab, kind) in enumerate(TESTS):
        g = cl[(cl.cohort == c) & (cl.test == t)]
        for j, gs in enumerate(GS_COL):
            s = g[g.gene_set == gs]
            if s.empty:
                continue
            s = s.iloc[0]
            y = i + (j - 1) * 0.22
            ax.scatter(s.pct_more_extreme_than_random * 100, y, s=12, color=GS_COL[gs], zorder=3, lw=0)
            rows.append(s.to_dict() | {"label": lab, "y": y})
        s = g[g.gene_set == "full"].iloc[0]
        stat = (f"AUROC {s.auroc:.2f} ({int(s.n_advanced)} vs {int(s.n_early)})" if kind == "auc"
                else f"ρ = {num(s.spearman)}, {fp(s.p)}, n = {int(s.n)}")
        ax.text(1.02, i, stat, ha="left", va="center", fontsize=5.5, transform=ax.get_yaxis_transform(),
                gid="free")
    ax.axvline(95, ls="--", color="black", lw=0.6)
    ax.axvline(50, ls=":", color="#8C8C8C", lw=0.6)
    ax.set_yticks(range(len(TESTS)), [x[2] for x in TESTS], fontsize=5.8)
    ax.set_ylim(len(TESTS) - 0.5, -0.6)
    ax.set_xlim(0, 104)
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=GS_COL[g], label=GS_LAB[g]) for g in GS_COL]
    ax.legend(handles=h, loc="upper left", fontsize=5.3)
    ax.set_xlabel("Percentile vs random gene sets (expected direction)")
    fl.save_source(NAME, "c", pd.DataFrame(rows))


def build():
    r = fl.read(FD + "fig3_replication_auroc.tsv")
    s = fl.read(FD + "fig3_kpmp_egfr_scatter.tsv")
    cl = fl.read(FD + "fig3_clinical.tsv")
    fig = fl.new_fig(fl.FULL_W, 135 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])
    a = fig.add_subplot(gs[0])
    r2 = gs[1].subgridspec(1, 2, width_ratios=[1, 1.45])
    b, c = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    for ax in (a, b, c):
        fl.reserve_label(ax)
    panel_a(a, r)
    panel_b(b, s, cl)
    panel_c(c, cl)
    return fl.finalize(fig, {"a": a, "b": b, "c": c})


if __name__ == "__main__":
    fl.run(build, NAME)
