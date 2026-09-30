"""Fig. 4 - adaptive and failed-repair tubular states in within-lineage tubular change (panels a-e).

Inputs (read only): results/20_repair_state/figdata/ (panel_a_*, panel_b_*, panel_c_shapley_blocks.tsv,
panel_d_*, panel_e_correlations.tsv, panel_g_clinical_tests.tsv, panel_h_key_genes.tsv) and, for panel c, the
17-factor Shapley decomposition that separates adaptive from failed-repair states
(results/21_robustness/A6_split/shapley_contrast.tsv, pt_tal_shares.tsv).
Panels d-e are drawn by fig4_repair_panels.py. Program reproducibility with contributing-gene expression
(panel_a1/panel_a2) and the reduced program in independent cohorts (fig4_repair_panels.panel_f) are drawn into
Fig. S8 (panels d-e); the bulk forest plot (figdata panel_f_*) is Fig. S8a-c.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402
import fig4_repair_panels as p2  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAME = "Fig4"
FD = "results/20_repair_state/figdata/"
PROGRAMS = [("PT", "aPT"), ("PT", "frPT"), ("PT", "rfPT"), ("PT", "dPT"), ("PT", "cycPT"),
            ("TAL", "aTAL"), ("TAL", "frTAL"), ("TAL", "rfTAL"), ("TAL", "dTAL")]
TYPES = [("PT_normal", "PT normal"), ("PT_repair", "PT aPT+frPT"), ("PT_other_alt", "PT degen./cycling"),
         ("TAL_normal", "TAL normal"), ("TAL_repair", "TAL aTAL+frTAL"), ("TAL_other_alt", "TAL degenerative"),
         ("LOH_rest", "Other LOH"), ("DCT", "DCT"), ("CNT_PC", "CNT/PC"), ("IC", "IC"), ("PODO", "Podocyte"),
         ("PEC", "PEC"), ("ENDO", "Endothelial"), ("ENDO_PT", "Peritubular EC"), ("STROMA", "Stromal"),
         ("MAC", "Macrophage"), ("TCELL", "T cell")]
REFERENCE_GENES = ["HAVCR1", "LCN2", "CDH6", "CD24", "SPP1"]   # canonical injury markers outside the SCP core
BLOCKS = [("composition", "Composition", "#55A868"),
          ("adaptive_fraction", "Adaptive fraction", "#DD8452"),
          ("adaptive_cell_profile", "Adaptive profile", "#F2C29E"),
          ("failed_repair_fraction", "Failed-repair fraction", "#C44E52"),
          ("failed_repair_cell_profile", "Failed-repair profile", "#E8A0A2"),
          ("PT_TAL_normal_cell_profile", "Normal PT/TAL profile", "#4C72B0"),
          ("PT_TAL_other", "Other PT/TAL states", "#9DB4D8"),
          ("other_lineage_state_mix", "Other lineages, state mix", "#CCB974"),
          ("other_lineage_profile", "Other lineages, profile", "#8C8C8C")]
OTHER_PT_TAL = ("PT_TAL_other_altered_profile", "PT_TAL_other_state_mix")   # degenerative/cycling profile + mix
CONTRASTS = [("snRNA", "CKD_vs_REF", "snRNA CKD"), ("scRNA", "CKD_vs_REF", "scRNA CKD"),
             ("snRNA", "CKD_perc_vs_REF_perc", "snRNA CKD (biopsy)"), ("snRNA", "DKD_vs_REF", "snRNA DKD"),
             ("snRNA", "HKD_vs_REF", "snRNA HKD"), ("snRNA", "AKI_vs_REF", "snRNA AKI")]
STATE_LABEL = {"rfPT": "aPT+frPT", "rfTAL": "aTAL+frTAL"}  # display names (data keys unchanged)
CATS = [("REF", "Healthy"), ("DKD", "DKD"), ("HKD", "HKD"), ("AKI", "AKI")]
CAT_COL = {"REF": "#8C8C8C", "DKD": "#C44E52", "HKD": "#937860", "AKI": "#DD8452", "OTHER": "#6A6A6A"}


def panel_a1(ax, ps, name=NAME):
    ps = ps.set_index(["lineage", "state"]).reindex(PROGRAMS).reset_index()
    y = np.arange(len(ps))
    rho = ps.lfc_spearman_sn_vs_sc
    ax.barh(y, rho, height=0.62, color=["#C44E52" if v < 0 else "#4C72B0" for v in rho], edgecolor="white", lw=0.3)
    ax.axvline(0, color="black", lw=0.5)
    for yi, (u, d) in enumerate(zip(ps.snRNA_n_up, ps.snRNA_n_down)):
        ax.text(1.0, yi, f"{u:.0f}/{d:.0f}", ha="left", va="center", fontsize=5)
    ax.text(1.0, -1.05, "up/down", ha="left", va="center", fontsize=5, fontstyle="italic")
    ax.set_yticks(y, [STATE_LABEL.get(s_, s_) for s_ in ps.state], fontsize=5.5)
    ax.set_ylim(len(ps) - 0.4, -1.6)
    ax.set_xlim(-0.45, 2.3)
    ax.set_xticks([0, 0.8], ["0", "0.8"])
    ax.set_xlabel("snRNA vs scRNA\nprogram log$_2$FC (ρ)")
    fl.save_source(name, "d_programs", ps)


def panel_a2(ax, se, genes, name=NAME):
    se = se[se.dataset == "snRNA"].set_index("gene").reindex(genes)
    m = se[[t for t, _ in TYPES]].T
    z = (m - m.mean(0)) / m.std(0).replace(0, np.nan)
    lim = 2.5
    im = ax.imshow(z.clip(-lim, lim).to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=5, fontstyle="italic")
    ax.set_yticks(range(len(TYPES)), [lab for _, lab in TYPES], fontsize=5)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    nref = sum(g in REFERENCE_GENES for g in genes)
    ax.axvline(len(genes) - nref - 0.5, color="black", lw=0.6)
    for yl in (2.5, 5.5):
        ax.axhline(yl, color="white", lw=0.8)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.6, aspect=10, pad=0.01)
    cb.set_label("z (mean log$_2$CPM)", fontsize=5, fontweight="bold")
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    fl.save_source(name, "d_heatmap", z.T.reset_index().rename(columns={"index": "gene"}).assign(
        reference_marker_outside_core=lambda t: t.gene.isin(REFERENCE_GENES)))


def panel_b(ax, ax_n, sc, st):
    rows = []
    for a, cells, title in ((ax, "all", "All PT cells"), (ax_n, "normal", "Normal PT")):
        d = sc[(sc.dataset == "snRNA") & (sc.lineage == "PT") & (sc.lineage_cells == cells)]
        bg = d[d.scp_core.isna()]
        a.scatter(bg.program_log2fc, bg.log2fc, s=0.6 if a is ax else 0.3, lw=0, color="#D0D0D0", rasterized=True)
        for dr, col in (("up", "#C44E52"), ("down", "#4C72B0")):
            s = d[d.scp_core == dr]
            a.scatter(s.program_log2fc, s.log2fc, s=2.2 if a is ax else 1.2, lw=0, color=col, alpha=0.8,
                      rasterized=True)
        a.axhline(0, color="#999999", lw=0.4)
        a.axvline(0, color="#999999", lw=0.4)
        s_core = st[(st.dataset == "snRNA") & (st.lineage == "PT") & (st.lineage_cells == cells) &
                    (st.contrast == "CKD_vs_REF") & (st.program == "rfPT")].set_index("gene_set")
        rc, pc = s_core.loc["scp_core", ["spearman", "p_perm"]]
        ra = s_core.loc["all_expressed", "spearman"]
        txt = f"{'response genes ' if a is ax else ''}ρ = {rc:.2f}\n{fl.fmt_p(pc)}" + (f"\nall genes ρ = {ra:.2f}" if a is ax else "")
        a.text(0.03, 0.97, txt, transform=a.transAxes, ha="left", va="top", fontsize=5 if a is ax else 4.8, zorder=5,
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
        a.set_title(title, fontsize=6, pad=2)
        a.set_xlim(-6, 8)
        a.set_ylim(-4, 5.5)
        rows.append(d.assign(panel_cells=cells, core_spearman=rc, core_p_perm=pc, all_spearman=ra))
    ax.set_xlabel("aPT+frPT program log$_2$FC")
    ax.set_ylabel("PT CKD vs healthy log$_2$FC")
    ax_n.set_xlabel("aPT+frPT")
    ax_n.tick_params(labelsize=5)
    h = [Line2D([], [], marker="o", ls="", ms=2.5, color=c, label=l) for c, l in
         (("#C44E52", "Response gene, up"), ("#4C72B0", "Response gene, down"), ("#D0D0D0", "Other genes"))]
    ax.legend(handles=h, loc="lower right", fontsize=4.8, handletextpad=0.1, borderpad=0.25)
    fl.save_source(NAME, "a", pd.concat(rows)[["lineage_cells", "gene", "program_log2fc", "log2fc", "p_perm",
                                              "scp_core", "core_spearman", "core_p_perm", "all_spearman"]])


def panel_c(ax, blk, sh):
    y = np.arange(len(CONTRASTS))
    rows = []
    blk = blk[blk.variant == "full"]
    sh = sh[sh.variant == "full"]
    for yi, (ds, ct, lab) in enumerate(CONTRASTS):
        g = blk[(blk.dataset == ds) & (blk.contrast == ct)].set_index("component")
        left = 0.0
        for comp, _, col in BLOCKS:
            parts = OTHER_PT_TAL if comp == "PT_TAL_other" else (comp,)
            v = float(sum(g.at[c, "share_of_full"] for c in parts))
            ax.barh(yi, v, left=left, height=0.66, color=col, edgecolor="white", lw=0.3)
            rows.append(dict(dataset=ds, contrast=ct, component=comp, source_components=";".join(parts),
                             share_of_full=v, n_A=g.at[parts[0], "n_A"], n_B=g.at[parts[0], "n_B"]))
            left += v
        t = sh[(sh.dataset == ds) & (sh.contrast == ct)].set_index("component")
        a, f = (float(t.at[c, "share_of_PT_TAL_state"]) for c in ("adaptive_total", "failed_repair_total"))
        ax.text(1.03, yi, f"{a * 100:.0f} / {f * 100:.0f}", ha="left", va="center", fontsize=5.5)
        for c, v in (("adaptive_share_of_PT_TAL_within_cell", a), ("failed_repair_share_of_PT_TAL_within_cell", f)):
            rows.append(dict(dataset=ds, contrast=ct, component=c, share_of_PT_TAL_within_cell=v))
    ax.text(1.03, -0.75, "Adaptive /\nfailed-\nrepair (%)", ha="left", va="bottom", fontsize=4.8)
    ax.set_yticks(y, [lab for _, _, lab in CONTRASTS], fontsize=5.5)
    ax.set_ylim(len(CONTRASTS) - 0.45, -2.6)
    ax.set_xlim(0, 1.36)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("% of injury–repair score difference (Shapley)")
    fl.save_source(NAME, "b", pd.DataFrame(rows))


def panel_c_key(ax):
    fl.hide(ax)
    h = [Patch(color=col, label=lab) for _, lab, col in BLOCKS]
    ax.legend(handles=h, loc="center left", fontsize=5, ncol=1, frameon=False, handlelength=0.9,
              borderaxespad=0)


def panel_d(ax, dt, ct):
    dt = dt[dt.dataset == "snRNA"]
    rows = []
    for li, (state, lab) in enumerate((("PT:rfPT", "aPT+frPT"), ("TAL:rfTAL", "aTAL+frTAL"))):
        for ci, (cat, clab) in enumerate(CATS):
            x = li * (len(CATS) + 1) + ci
            v = dt[dt.category == cat][state].dropna() * 100
            ax.boxplot([v], positions=[x], widths=0.6, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=CAT_COL[cat], lw=0.6),
                       medianprops=dict(color=CAT_COL[cat], lw=1.0), whiskerprops=dict(color=CAT_COL[cat], lw=0.5),
                       capprops=dict(color=CAT_COL[cat], lw=0.5))
            ax.scatter(x + fl.jitter(len(v), 0.2), v, s=2, lw=0, color=CAT_COL[cat], alpha=0.7, zorder=3)
            t = ct[(ct.dataset == "snRNA") & (ct.state == state) & (ct.group_A == cat) & (ct.group_B == "REF")]
            rows.append(pd.DataFrame({"state": state, "category": cat, "donor_pct": v.values,
                                      "median_pct": v.median(),
                                      "hedges_g_vs_REF": t.hedges_g.iloc[0] if len(t) else np.nan,
                                      "p_perm_vs_REF": t.p_perm.iloc[0] if len(t) else np.nan}))
        ax.text(li * (len(CATS) + 1) + 1.5, 86, lab, ha="center", va="center", fontsize=5.5)
    ax.set_xticks([li * (len(CATS) + 1) + ci for li in range(2) for ci in range(len(CATS))],
                  [c for _ in range(2) for _, c in CATS], rotation=90, fontsize=5.5)
    ax.set_xlim(-0.7, 2 * len(CATS) + 0.7)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Adaptive + failed-repair\ncells (% of lineage)")
    fl.save_source(NAME, "c", pd.concat(rows))


def build():
    kg = fl.read(FD + "panel_h_key_genes.tsv")
    sc = fl.read(FD + "panel_b_concordance_scatter.tsv.gz")
    st = fl.read(FD + "panel_b_concordance_stats.tsv")
    blk = fl.read("results/21_robustness/A6_split/shapley_contrast.tsv")
    sh = fl.read("results/21_robustness/A6_split/pt_tal_shares.tsv")
    dt = fl.read(FD + "panel_d_donor_table.tsv", dtype={"donor": str})
    ct = fl.read(FD + "panel_d_category_tests.tsv")
    cor = fl.read(FD + "panel_e_correlations.tsv")

    fig = fl.new_fig(fl.FULL_W, 140 * fl.MM)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
    r2 = gs[0].subgridspec(1, 5, width_ratios=[1.25, 0.62, 1.55, 0.75, 1.3])
    b, bn = fig.add_subplot(r2[0]), fig.add_subplot(r2[1])
    c, ckey = fig.add_subplot(r2[2]), fig.add_subplot(r2[3])
    d = fig.add_subplot(r2[4])
    r3 = gs[1].subgridspec(1, 3, width_ratios=[1.25, 0.65, 2.2])
    e, en = fig.add_subplot(r3[0]), fig.add_subplot(r3[1])
    g = fig.add_subplot(r3[2])
    for ax in (b, c, d, e, g):
        fl.reserve_label(ax)
    panel_b(b, bn, sc, st)
    panel_c(c, blk, sh)
    panel_c_key(ckey)
    panel_d(d, dt, ct)
    p2.panel_e(e, en, dt, cor, NAME)
    p2.panel_g(g, kg, NAME)
    return fl.finalize(fig, {"a": b, "b": c, "c": d, "d": e, "e": g})


if __name__ == "__main__":
    fl.run(build, NAME)
