"""Graphical abstract (landscape 920 x 300 px; PNG + SVG, plus a 300-dpi PNG) drawn with code.

Numbers are read from result tables only:
manuscript/numbers/{resource,signature_audit}.tsv, results/19_shared_program/figdata/{key_numbers,
fig2_kpmp_decomposition,fig3_replication_auroc,fig3_clinical}.tsv, results/14_kpmp/claims.tsv,
results/19_shared_program/figdata/fig4_residual_tests.tsv, results/21_robustness/A6_split/pt_tal_shares.tsv.
Drawing coordinates are pixels of the 920 x 300 canvas. Texts drawn -> figures/source_data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
from Fig1_framework import tint  # noqa: E402

NAME = "graphical_abstract"
W, H = 920, 300
COLS = [fl.PALETTE[0], fl.PALETTE[3], fl.PALETTE[2], fl.PALETTE[4], fl.PALETTE[1], fl.PALETTE[5]]


def numbers() -> dict:
    res = fl.kv("manuscript/numbers/resource.tsv")
    sig = fl.kv("manuscript/numbers/signature_audit.tsv")  # INJURY rows: shared-program score AUROC range
    key = fl.kv("results/19_shared_program/figdata/key_numbers.tsv")
    R = "results/19_shared_program/figdata/"
    dec = fl.read(R + "fig2_kpmp_decomposition.tsv")
    t = dec[(dec.dataset == "snRNA") & (dec.compartment == "TUB") & (dec.contrast == "CKD_vs_REF")]
    d = t.set_index("kind")["delta"]
    rep = fl.read(R + "fig3_replication_auroc.tsv")
    import yaml
    cfg = yaml.safe_load((fl.ROOT / "config" / "run.yaml").read_text())
    indep = list(cfg["shared_program"]["replicate"]["cohorts"])  # the six cohorts never used as sources
    rep = rep[rep.primary & (rep.gene_set == "full") & rep.cohort.isin(indep)]
    cl = fl.read(R + "fig3_clinical.tsv")
    eg = cl[(cl.cohort == "KPMP_snRNA_donors") & (cl.core == "TUB") & (cl.gene_set == "full")
            & (cl.test == "egfr_bin_CKD")].iloc[0]
    sel = (cl.core == "TUB") & (cl.gene_set == "full")
    eg_dkd = cl[sel & (cl.cohort == "KPMP_snRNA_donors") & (cl.test == "egfr_bin_DKD")].iloc[0]
    prot = cl[sel & (cl.cohort == "KPMP") & (cl.test == "proteinuria_cat_patients")].iloc[0]
    stg = cl[sel & (cl.cohort == "GSE142025-RNAseq") & (cl.test == "advanced_vs_early_DN")].iloc[0]
    cla = fl.read("results/14_kpmp/claims.tsv")
    podo = cla[(cla.contrast == "DKD_vs_PAT_adj") & (cla.cell_type == "PODO") & (cla.unit == "META")].iloc[0]
    sh = fl.read("results/21_robustness/A6_split/pt_tal_shares.tsv")
    sh = sh[(sh.dataset == "snRNA") & (sh.contrast == "CKD_vs_REF") & (sh.variant == "full")].set_index("component")
    rt = fl.read(R + "fig4_residual_tests.tsv")
    r2 = [float(v) for k, v in key.items() if k.startswith("composition_r2cv_median_")]
    return dict(
        n_samples=f"{int(res['n_samples_kept']):,}", n_spec=f"{int(res['n_unique_specimens']):,}",
        n_cohorts=res["n_evaluation_cohorts"],
        auc_lo=float(sig["INJURY_auc_dkd_ctrl_min"]), auc_hi=float(sig["INJURY_auc_dkd_ctrl_max"]),
        core_g=int(float(key["n_core_GLOM_up"]) + float(key["n_core_GLOM_down"])),
        core_t=int(float(key["n_core_TUB_up"]) + float(key["n_core_TUB_down"])),
        r2_lo=min(r2), r2_hi=max(r2),
        state=d["state_only"] / d["full"], comp=d["comp_only"] / d["full"],
        n_rep=len(rep), rep_lo=rep.auroc.min(), rep_hi=rep.auroc.max(), egfr_rho=eg.spearman,
        podo_z=podo.z_snRNA, adaptive=float(sh.at["adaptive_total", "share_of_PT_TAL_state"]),
        failed=float(sh.at["failed_repair_total", "share_of_PT_TAL_state"]),
        n_rep_above=int((rep.pct_vs_random >= 0.95).sum()),
        rep_dx=sorted(set(rt[rt.status == "replicated"].disease)),
        egfr_rho_dkd=eg_dkd.spearman, prot_rho=prot.spearman, stage_auc=stg.auroc)


def columns(N: dict) -> list[tuple[str, list[str]]]:
    m = "−"
    return [
        ("Specimens", [f"{N['n_samples']} samples", f"\u2192 {N['n_spec']} unique", "biopsy specimens",
                       "counted once", f"{N['n_cohorts']} cohorts"]),
        ("Injury–repair response", ["learned without DKD", "DKD vs control", f"AUROC {N['auc_lo']:.3f}\u2013{N['auc_hi']:.3f}",
                            f"{N['core_g']:,} + {N['core_t']:,}", "response genes"]),
        ("Key cells", ["KPMP tubular signal:", f"within lineages {N['state'] * 100:.0f}%", f"cell proportions {N['comp'] * 100:.0f}%",
                       "adaptive / failed-repair:", f"{N['adaptive'] * 100:.0f}% / {N['failed'] * 100:.0f}% of PT/TAL"]),
        ("Replication", [f"{N['n_rep']} independent cohorts", "disease vs healthy", f"AUROC {N['rep_lo']:.2f}\u2013{N['rep_hi']:.2f}",
                         f"{N['n_rep_above']} of {N['n_rep']} above", "random gene sets"]),
        ("Kidney function", ["KPMP donor eGFR", f"CKD \u03c1 = {N['egfr_rho']:.2f}".replace("-", m),
                             f"DKD \u03c1 = {N['egfr_rho_dkd']:.2f}".replace("-", m),
                             f"proteinuria \u03c1 = {N['prot_rho']:.2f}", f"DKD stage AUROC {N['stage_auc']:.2f}"]),
        ("Disease-specific", ["small after", "adjustment", "DKD: podocyte loss", f"(z {N['podo_z']:.1f})".replace("-", m),
                               "replicated signature:", " ".join(N["rep_dx"]) + " only"]),
    ]


def build():
    N = numbers()
    fig = plt.figure(figsize=((W + 0.4) / 100, (H + 0.4) / 100), dpi=100)  # +0.4 px: float truncation
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    rows = []
    ax.add_patch(FancyBboxPatch((10, 262), W - 20, 30, boxstyle="round,pad=0,rounding_size=6", lw=0,
                                fc="#3A3A3A", gid="box:title"))
    title = "A shared injury–repair response involving adaptive and failed-repair tubular states"
    ax.text(W / 2, 277, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white",
            gid="in:title")
    rows.append(dict(element="title", text=title))
    cols = columns(N)
    n, gap, x0 = len(cols), 14, 10
    cw = (W - 2 * x0 - (n - 1) * gap) / n
    top, bot = 250, 52
    for j, ((head, lines), c) in enumerate(zip(cols, COLS)):
        x = x0 + j * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, bot), cw, top - bot, boxstyle="round,pad=0,rounding_size=8", lw=1.0,
                                    ec=c, fc=tint(c, 0.12), gid=f"box:c{j}", zorder=1))
        ax.add_patch(FancyBboxPatch((x, top - 30), cw, 30, boxstyle="round,pad=0,rounding_size=8", lw=0,
                                    fc=c, gid=f"box:h{j}", zorder=2))
        ax.text(x + cw / 2, top - 15, head, ha="center", va="center", fontsize=8.8 if len(head) < 20 else 8.2, fontweight="bold",
                color="white", gid=f"in:h{j}", zorder=3)
        ax.text(x + cw / 2, (bot + top - 30) / 2, "\n".join(lines), ha="center", va="center", fontsize=8.5,
                linespacing=1.5, gid=f"in:c{j}", zorder=3)
        rows.append(dict(element=f"column{j + 1}", text=head + ": " + " | ".join(lines)))
        if j < n - 1:
            ya = (bot + top) / 2
            ax.add_patch(FancyArrowPatch((x + cw + 1.5, ya), (x + cw + gap - 1.5, ya), arrowstyle="-|>",
                                         mutation_scale=9, lw=1.0, color="#555555", zorder=2))
    msg = "Disease-specific signals are small beside the injury–repair response: compare patients with patients"
    ax.add_patch(FancyBboxPatch((10, 10), W - 20, 32, boxstyle="round,pad=0,rounding_size=6", lw=0.8,
                                ec="#3A3A3A", fc="#F2F2F2", gid="box:msg"))
    ax.text(W / 2, 26, msg, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#3A3A3A",
            gid="in:msg")
    rows.append(dict(element="takeaway", text=msg))
    fl.save_source(NAME, "all", pd.DataFrame(rows))
    return fig


def main() -> None:
    fig = build()
    fl.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(fl.FIG_DIR / f"{NAME}.png", dpi=100)
    fig.savefig(fl.FIG_DIR / f"{NAME}.svg")
    fig.savefig(fl.FIG_DIR / f"{NAME}_hires.png", dpi=300)
    print("wrote figures/graphical_abstract.{png,svg} (920 x 300 px) and graphical_abstract_hires.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
