"""Panels e-g of Fig. 4 (failed-repair tubular states): donor fractions vs SCP and eGFR, independent clinical
cohorts, key driver genes. Plotting only; every value is read from results/20_repair_state/figdata/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as fl
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

CAT_COL = {"REF": "#8C8C8C", "DKD": "#C44E52", "HKD": "#937860", "OTHER": "#6A6A6A"}
CAT_LAB = {"REF": "Healthy", "DKD": "DKD", "HKD": "HKD", "OTHER": "Other CKD"}
PROG = "prog_noSCP_PT:rfPT"          # rfPT program with every SCP core gene removed
CLIN_RHO = [("GSE175759-RNAseq", "egfr_patients", "GSE175759 eGFR"),
            ("GSE115857-GPL14951", "iga_grade_G1_G3", "GSE115857 IgAN grade"),
            ("KPMP", "egfr_bin_patients", "KPMP sections eGFR"),
            ("KPMP", "proteinuria_cat_patients", "KPMP sections proteinuria")]
MK = {"raw": dict(marker="o", color="#C44E52", mfc="#C44E52", label="Program"),
      "partial": dict(marker="o", color="#C44E52", mfc="white", label="Program, adjusted"),
      "scp": dict(marker="D", color="#8C8C8C", mfc="#8C8C8C", label="Injury–repair score")}
STATES = ["aPT", "frPT", "aTAL", "frTAL"]
N_KEY = 12
REF_MARKERS = ["HAVCR1", "LCN2"]


def _rho(cor, donors, target):
    r = cor[(cor.dataset == "snRNA") & (cor.donors == donors) & (cor.variable == "PT:rfPT") & (cor.target == target)]
    return r.iloc[0]


def panel_e(ax, ax_n, dt, cor, name):
    dt = dt[(dt.dataset == "snRNA") & dt.category.isin(list(CAT_COL))]
    for cat, col in CAT_COL.items():
        s = dt[dt.category == cat]
        ax.scatter(s["PT:rfPT"] * 100, s.scp_full, s=4, lw=0, color=col, alpha=0.85, label=CAT_LAB[cat])
    r_all, r_ckd = _rho(cor, "REF+CKD", "scp_full"), _rho(cor, "CKD", "scp_full")
    ax.text(0.97, 0.03, f"all ρ = {r_all.spearman:.2f}\nCKD ρ = {r_ckd.spearman:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5)
    ax.set_xlabel("aPT+frPT (% of PT)")
    ax.set_ylabel("Donor injury–repair score")
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.15, 0.2)
    ax.legend(loc="upper left", fontsize=4.8, handletextpad=0.1, borderpad=0.25, markerscale=1.2)
    ck = dt[dt.category.isin(["DKD", "HKD", "OTHER"]) & dt.egfr_mid.notna()]
    for cat in ("DKD", "HKD", "OTHER"):
        s = ck[ck.category == cat]
        ax_n.scatter(s["PT:rfPT"] * 100, s.egfr_mid, s=3, lw=0, color=CAT_COL[cat], alpha=0.85)
    r_e = _rho(cor, "CKD", "egfr_mid")
    ax_n.text(0.97, 0.97, f"ρ = {r_e.spearman:.2f}\n{fl.fmt_p(r_e.p_perm)}".replace("-", "−"),
              transform=ax_n.transAxes, ha="right", va="top", fontsize=4.6)
    ax_n.set_title("CKD", fontsize=6, pad=2)
    ax_n.set_xlabel("aPT+frPT (%)")
    ax_n.set_ylabel("eGFR", labelpad=1)
    ax_n.set_xlim(0, 75)
    ax_n.set_ylim(0, 150)
    ax_n.tick_params(labelsize=5)
    stats = pd.DataFrame([r_all, r_ckd, r_e])
    fl.save_source(name, "d", dt[["donor", "category", "PT:rfPT", "scp_full", "egfr_mid"]].assign(
        rho_all_scp=r_all.spearman, p_all_scp=r_all.p_perm, rho_ckd_scp=r_ckd.spearman, p_ckd_scp=r_ckd.p_perm,
        rho_ckd_egfr=r_e.spearman, p_ckd_egfr=r_e.p_perm, n_ckd_egfr=r_e.n))
    fl.save_source(name, "d_stats", stats)


def panel_f(ax, ax2, clin, name):
    c = clin[clin.samples == "all"]
    rows = []
    for yi, (coh, test, lab) in enumerate(CLIN_RHO):
        p = c[(c.cohort == coh) & (c.test == test) & (c.measure == PROG)].iloc[0]
        s = c[(c.cohort == coh) & (c.test == test) & (c.measure == "scp")].iloc[0]
        vals = {"raw": p.spearman, "partial": p.partial_spearman_given_scp, "scp": s.spearman}
        for k, off in (("raw", -0.2), ("partial", 0.0), ("scp", 0.2)):
            m = MK[k]
            ax.plot([vals[k]], [yi + off], ls="", marker=m["marker"], ms=3.2, mec=m["color"], mfc=m["mfc"], mew=0.7)
        rows.append(dict(cohort=coh, test=test, n=p.n, rho_program=p.spearman, p_program=p.p_perm,
                         partial_rho_given_scp=p.partial_spearman_given_scp, p_partial=p.p_perm_partial,
                         rho_scp=s.spearman, p_scp=s.p_perm))
    ax.axhspan(1.5, 3.5, color="#F0F0F0", zorder=0, lw=0)
    ax.text(0.03, 3.45, "not reproduced", ha="left", va="bottom", fontsize=4.8, fontstyle="italic",
            transform=ax.get_yaxis_transform())
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(CLIN_RHO)), [lab.replace(" ", "\n", 1) for _, _, lab in CLIN_RHO], fontsize=5)
    ax.set_ylim(len(CLIN_RHO) - 0.45, -2.2)
    ax.set_xlim(-0.6, 0.8)
    ax.set_xticks([-0.4, 0, 0.4, 0.8], ["−0.4", "0", "0.4", "0.8"])
    ax.set_xlabel("Spearman ρ")
    h = [Line2D([], [], ls="", marker=MK[k]["marker"], ms=3, mec=MK[k]["color"], mfc=MK[k]["mfc"], mew=0.7,
                label=MK[k]["label"]) for k in MK]
    ax.legend(handles=h, loc="upper left", fontsize=4.8, handletextpad=0.1, borderpad=0.25, labelspacing=0.2)
    p = c[(c.cohort == "GSE142025-RNAseq") & (c.test == "advanced_vs_early_DN") & (c.measure == PROG)].iloc[0]
    s = c[(c.cohort == "GSE142025-RNAseq") & (c.test == "advanced_vs_early_DN") & (c.measure == "scp")].iloc[0]
    vals = {"raw": p.auroc, "partial": p.auroc_given_scp_resid, "scp": s.auroc}
    for k, off in (("raw", -0.2), ("partial", 0.0), ("scp", 0.2)):
        m = MK[k]
        ax2.plot([vals[k]], [off], ls="", marker=m["marker"], ms=3.2, mec=m["color"], mfc=m["mfc"], mew=0.7)
    ax2.axvline(0.5, color="black", lw=0.5, ls="--")
    ax2.set_yticks([0], ["GSE142025\nadvanced\nvs early"], fontsize=5)
    ax2.set_ylim(0.6, -0.6)
    ax2.set_xlim(0.3, 1.05)
    ax2.set_xticks([0.4, 0.7, 1.0])
    ax2.set_xlabel("AUROC")
    ax2.tick_params(labelsize=5)
    rows.append(dict(cohort="GSE142025-RNAseq", test="advanced_vs_early_DN", n=p.n, auroc_program=p.auroc,
                     auroc_program_given_scp_resid=p.auroc_given_scp_resid, auroc_scp=s.auroc))
    fl.save_source(name, "e", pd.DataFrame(rows))


def panel_g(ax, kg, name):
    top = kg[kg.is_driver & (kg.scp_direction == "up")].head(N_KEY)
    ref = kg[kg.gene.isin(REF_MARKERS)]
    t = pd.concat([top, ref]).reset_index(drop=True)
    y = np.arange(len(t))
    lim = 4.0
    rows = []
    for j, st in enumerate(STATES):
        lfc = t[f"snRNA_log2fc_{st}"]
        fdr = t[f"snRNA_fdr_{st}"]
        drv = t.driver_of.fillna("").str.split(";").apply(lambda s, st=st: st in s)
        size = np.clip(-np.log10(fdr.clip(lower=1e-30)), 0, 30) / 30 * 22 + 2
        ax.scatter(np.full(len(t), j), y, s=size, c=lfc.clip(-lim, lim), cmap="RdBu_r",
                   norm=TwoSlopeNorm(0, -lim, lim), edgecolors=np.where(drv, "black", "none"), linewidths=0.5,
                   zorder=3)
        rows.append(pd.DataFrame({"gene": t.gene, "state": st, "snRNA_log2fc": lfc, "snRNA_fdr": fdr,
                                  "driver_call": drv}))
    x0, w = 4.6, 2.6
    for yi, r in t.iterrows():
        share = r.tissue_share_repair_attributable
        if np.isfinite(share):
            ax.barh(yi, share * w, left=x0, height=0.6, color="#C44E52", edgecolor="white", lw=0.3)
            ax.text(x0 + share * w + 0.08, yi, f"{share * 100:.0f}%", ha="left", va="center", fontsize=4.8)
        else:
            ax.text(x0 + 0.1, yi, "non-response", ha="left", va="center", fontsize=4.8, fontstyle="italic",
                    color="#6A6A6A")
        ax.text(x0 + w + 1.6, yi, f"{r.n_indep_cohorts_concordant:.0f}/{r.n_indep_cohorts_tested:.0f}",
                ha="left", va="center", fontsize=5)
    ax.plot([x0, x0], [-0.5, len(t) - 0.5], color="black", lw=0.5)
    ax.axhline(len(top) - 0.5, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(list(range(len(STATES))) + [x0 + w / 2, x0 + w + 1.85],
                  [f"{s_} program" for s_ in STATES] + ["aPT+frPT share\nof tissue Δ", "Indep.\ncohorts"],
                  fontsize=5, rotation=90)
    ax.set_yticks(y, t.gene, fontsize=5.5, fontstyle="italic")
    ax.set_ylim(len(t) - 0.4, -0.7)
    ax.set_xlim(-0.6, x0 + w + 2.6)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    sm = ax.collections[0]
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.01)
    cb.set_label("program log$_2$FC", fontsize=5, fontweight="bold")
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    keep = ["gene", "in_scp_core_TUB", "scp_direction", "driver_of", "tissue_delta_full",
            "tissue_delta_repair_attributable", "tissue_share_repair_attributable", "tissue_share_normal_cell_profile",
            "n_indep_cohorts_concordant", "n_indep_cohorts_tested", "bulk_meta_g_re", "bulk_meta_fdr"]
    fl.save_source(name, "e", t[keep].merge(pd.concat(rows).pivot_table(
        index="gene", columns="state", values="snRNA_log2fc").add_prefix("snRNA_log2fc_").reset_index(), on="gene"))
    fl.save_source(name, "e_dots", pd.concat(rows))
