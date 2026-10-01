"""Fig. 1 - study overview (code-drawn vector schematic, panels a-f).

Every number printed in the schematic is read from a results/config table:
manuscript/numbers/resource.tsv, config/run.yaml, results/14_kpmp/claims.tsv,
results/19_shared_program/figdata/fig4_residual_tests.tsv,
results/19_shared_program/figdata/{key_numbers,fig2_kpmp_decomposition}.tsv,
results/20_repair_state/programs/program_summary.tsv.
The texts actually drawn are written to figures/source_data/Fig1_panel<x>.tsv.
Drawing coordinates are millimetres (each panel axes spans its own size in mm, aspect 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

NAME = "Fig1"
FS = 6.2          # body font size (pt)
GAP = 2.4         # vertical gap between boxes (mm), arrows live here
ACCENT = {"a": fl.PALETTE[0], "b": fl.PALETTE[3], "c": fl.PALETTE[2], "d": fl.PALETTE[4],
          "e": fl.PALETTE[1], "f": fl.PALETTE[5]}
TITLES = {"a": "Unique biopsy specimens", "b": "Injury–repair response",
          "c": "Compositional and within-lineage change", "d": "Adaptive and failed-repair states",
          "e": "Replication and kidney function", "f": "Disease-specific signals"}
RECORDS: dict[str, list] = {}


# ---------------------------------------------------------------- drawing helpers (mm coordinates)
def tint(color: str, f: float) -> tuple:
    """Mix a colour with white (f = share of the colour)."""
    c = mcolors.to_rgb(color)
    return tuple(1 - f * (1 - x) for x in c)


def box(ax, key: str, x: float, y: float, w: float, h: float, text: str, accent: str, fs: float = FS,
        weight: str = "normal", fill: float = 0.14, edge: bool = True, color: str = "black", panel: str = ""):
    """Rounded box with centred text; text gid 'in:<key>' / patch gid 'box:<key>' for the checker."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", lw=0.6,
                       fc=tint(accent, fill), ec=accent if edge else "none", gid=f"box:{key}", zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight,
            color=color, linespacing=1.25, gid=f"in:{key}", zorder=3)
    if panel:
        RECORDS.setdefault(panel, []).append(dict(box=key, x_mm=x, y_mm=y, w_mm=w, h_mm=h,
                                                  text=text.replace("\n", " | ")))
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1, color="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 color=color, shrinkA=0.4, shrinkB=0.4, zorder=1))


def down(ax, upper, lower, xfrac_u=0.5, xfrac_l=0.5):
    """Arrow from the bottom of box `upper` to the top of box `lower` (boxes as (x, y, w, h))."""
    arrow(ax, upper[0] + upper[2] * xfrac_u, upper[1], lower[0] + lower[2] * xfrac_l, lower[1] + lower[3])


def stack(top: float, heights: list[float], gap: float = GAP) -> list[float]:
    """y (bottom) of boxes stacked downwards from `top`."""
    ys, y = [], top
    for h in heights:
        y -= h
        ys.append(y)
        y -= gap
    assert ys[-1] >= 0.5, f"boxes overflow the panel bottom ({ys[-1]:.1f} mm)"
    return ys


def panel_axes(fig, rect_mm, W, H):
    x, y, w, h = rect_mm
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])  # no tick labels at all (axis-off axes still carry tick Text objects)
    ax.set_axis_off()
    return ax


def header(ax, k: str, w: float, h: float) -> float:
    """Coloured title band across the top of a panel; returns the y below it."""
    box(ax, f"{k}_title", 0, h - 6.0, w, 6.0, TITLES[k], ACCENT[k], fs=7, weight="bold", fill=0.85,
        edge=False, color="white", panel=k)
    return h - 6.0 - GAP


# ---------------------------------------------------------------- inputs
def load_numbers() -> dict:
    res = fl.kv("manuscript/numbers/resource.tsv")
    key = fl.kv("results/19_shared_program/figdata/key_numbers.tsv")
    cfg_p = fl.ROOT / "config" / "run.yaml"
    fl.INPUTS.append(cfg_p)
    cfg = yaml.safe_load(cfg_p.read_text())
    dec = fl.read("results/19_shared_program/figdata/fig2_kpmp_decomposition.tsv")
    full = dec[(dec.kind == "full") & (dec.contrast == "CKD_vs_REF") & (dec.compartment == "TUB")]
    donors = {r.dataset: (int(r.n_A), int(r.n_B)) for r in full.itertuples()}
    cla = fl.read("results/14_kpmp/claims.tsv")
    podo = cla[(cla.contrast == "DKD_vs_PAT_adj") & (cla.cell_type == "PODO") & (cla.unit == "META")].iloc[0]
    rt = fl.read("results/19_shared_program/figdata/fig4_residual_tests.tsv")
    dk = rt[(rt.disease == "DKD") & (rt.status != "descriptive")]
    ps = fl.read("results/20_repair_state/programs/program_summary.tsv")
    rf = ps[(ps.lineage == "PT") & (ps.state == "rfPT")].iloc[0]
    return dict(res=res, key=key, cfg=cfg, donors=donors, podo_z=podo.z_snRNA, rf_donors=int(rf.snRNA_n_donors),
                rep_dx=sorted(set(rt[rt.status == "replicated"].disease)), n_dkd_tested=len(dk),
                n_dkd_notrep=int((dk.status == "not_replicated").sum()))


def i(s: str) -> str:
    """'1462' / '1462.0' -> '1,462'."""
    return f"{int(float(s)):,}"


# ---------------------------------------------------------------- panels
def panel_a(ax, w, h, N):
    k, c = "a", ACCENT["a"]
    r, ov = N["res"], N["cfg"]["overlap"]
    top = header(ax, k, w, h)
    hs = [8, 8, 11, 6.5, 11]
    ys = stack(top, hs)
    b1 = box(ax, "a1", 1, ys[0], w - 2, hs[0], f"{r['n_series_versions_ingested']} GEO datasets\n"
             f"{r['n_series_versions_kept']} kept, {r['excluded_series'].split('-')[0]} excluded", c,
             panel=k)
    b2 = box(ax, "a2", 1, ys[1], w - 2, hs[1], f"{i(r['n_samples_kept'])} samples, {r['n_gse_kept']} GEO series, "
             f"{r['n_platforms_kept']} platforms\n{r['n_rnaseq_series']} RNA-seq series, "
             f"{r['n_array_series_versions']} array datasets", c, panel=k)
    b3 = box(ax, "a3", 1, ys[2], w - 2, hs[2], "Copies of a specimen across series:\nspecimen identifiers + residual "
             f"correlation\n({ov['resid_pcs']} PCs removed; r ≥ {ov['resid_min']}, "
             f"{ov['resid_min_within']} within series)", c, panel=k)
    b4 = box(ax, "a4", 1, ys[3], w - 2, hs[3], f"{i(r['n_unique_specimens'])} unique specimens "
             f"({i(r['n_duplicate_edges'])} duplicate links)", c, weight="bold", fill=0.3, panel=k)
    lw_ = (w - 4) * 0.42
    b5 = box(ax, "a5", 1, ys[4], lw_, hs[4], f"{r['n_evaluation_cohorts']} cohorts sharing\nno specimen", c, panel=k)
    b6 = box(ax, "a6", 1 + lw_ + 2, ys[4], w - 4 - lw_, hs[4], "sc/snRNA references\nGSE131882, GSE209781\n"
             "KPMP atlas v2", fl.PALETTE[9], panel=k)
    for u, l in ((b1, b2), (b2, b3), (b3, b4)):
        down(ax, u, l)
    down(ax, b4, b5, (lw_ / 2) / (w - 2), 0.5)


def panel_b(ax, w, h, N):
    k, c = "b", ACCENT["b"]
    inj, core = N["cfg"]["injury"], N["cfg"]["shared_program"]["core"]
    kn = N["key"]
    top = header(ax, k, w, h)
    hs = [8, 8, 8, 11, 8]
    ys = stack(top, hs)
    b1 = box(ax, "b1", 1, ys[0], w - 2, hs[0], "Non-DKD patients vs controls in source\ncohorts "
             "(DKD samples excluded)", c, panel=k)
    b2 = box(ax, "b2", 1, ys[1], w - 2, hs[1], "External score (non-DKD patients, other cohorts):\n"
             f"top {inj['top_k']} up − top {inj['top_k']} down genes", c, panel=k)
    b3 = box(ax, "b3", 1, ys[2], w - 2, hs[2], "Diagnosis changes vs injury–repair response:\ncosine, "
             "cross-cohort similarity, split controls", c, panel=k)
    b4 = box(ax, "b4", 1, ys[3], w - 2, hs[3], "Random-effects meta-analysis (Hartung–Knapp)\n"
             f"FDR < {core['fdr']}, |g| ≥ {core['min_abs_g']}, same sign in all sources\n"
             f"leave-one-out P < {core['loo_max_p']} and |g| ≥ {core['loo_min_abs_g']}", c, panel=k)
    ng = int(float(kn["n_core_GLOM_up"])) + int(float(kn["n_core_GLOM_down"]))
    nt = int(float(kn["n_core_TUB_up"])) + int(float(kn["n_core_TUB_down"]))
    b5 = box(ax, "b5", 1, ys[4], w - 2, hs[4], f"Glomerulus: {ng:,} genes ({i(kn['n_core_GLOM_up'])} up, "
             f"{i(kn['n_core_GLOM_down'])} down)\nTubulointerstitium: {nt:,} genes ({i(kn['n_core_TUB_up'])} up, "
             f"{i(kn['n_core_TUB_down'])} down)", c, weight="bold", fill=0.3, panel=k)
    for u, l in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        down(ax, u, l)


def panel_c(ax, w, h, N):
    k, c = "c", ACCENT["c"]
    cfg = N["cfg"]
    comp = cfg["shared_program"]["composition"]
    (sn_a, sn_b), (sc_a, sc_b) = N["donors"]["snRNA"], N["donors"]["scRNA"]
    top = header(ax, k, w, h)
    hs = [8, 8, 8, 11, 8]
    ys = stack(top, hs)
    b1 = box(ax, "c1", 1, ys[0], w - 2, hs[0], "BayesPrism deconvolution, KPMP atlas v2\nreference "
             f"({cfg['deconv']['primary']} primary; snRNA sensitivity)", c, panel=k)
    b2 = box(ax, "c2", 1, ys[1], w - 2, hs[1], "Bulk cohorts: CLR composition → injury–repair score\n"
             f"{comp['cv_folds']}-fold cross-validated R² per cohort", c, panel=k)
    b3 = box(ax, "c3", 1, ys[2], w - 2, hs[2], f"KPMP donors: snRNA {sn_a} CKD, {sn_b} reference\n"
             f"scRNA {sc_a} CKD, {sc_b} reference", c, panel=k)
    half = (w - 2 - 2) / 2
    b4 = box(ax, "c4", 1, ys[3], half, hs[3], "Compositional only\ndonor fractions ×\nreference profiles",
             c, panel=k)
    b5 = box(ax, "c5", 1 + half + 2, ys[3], half, hs[3], "Within-lineage only\nreference fractions ×\ndonor "
             "profiles", c, panel=k)
    b6 = box(ax, "c6", 1, ys[4], w - 2, hs[4], "Share of the response: compositional,\n"
             "within-lineage and their interaction", c, weight="bold", fill=0.3, panel=k)
    down(ax, b1, b2)
    down(ax, b2, b3)
    fq = (half / 2) / (w - 2)
    down(ax, b3, b4, fq, 0.5)
    down(ax, b3, b5, 1 - fq, 0.5)
    down(ax, b4, b6, 0.5, fq)
    down(ax, b5, b6, 0.5, 1 - fq)


def panel_d(ax, w, h, N):
    k, c = "d", ACCENT["d"]
    rs = N["cfg"]["repair_state"]
    top = header(ax, k, w, h)
    hs = [8, 11, 8, 8, 8]
    ys = stack(top, hs)
    b1 = box(ax, "d1", 1, ys[0], w - 2, hs[0], "KPMP PT and TAL states: normal, adaptive,\n"
             "failed-repair, degenerative, cycling", c, panel=k)
    b2 = box(ax, "d2", 1, ys[1], w - 2, hs[1], "State programs within donors, no disease labels\n"
             f"(state vs normal cells; ≥ {rs['min_cells']} cells per donor and state;\n"
             f"aPT+frPT: {N['rf_donors']} snRNA donors, scRNA replication)", c, panel=k)
    b3 = box(ax, "d3", 1, ys[2], w - 2, hs[2], "Shapley decomposition, 17 factors: adaptive\n"
             "and failed-repair fraction and profile", c, panel=k)
    b4 = box(ax, "d4", 1, ys[3], w - 2, hs[3], "State fractions vs injury–repair score and\n"
             "eGFR (partial correlation given the score)", c, panel=k)
    b5 = box(ax, "d5", 1, ys[4], w - 2, hs[4], "Contributing genes: program effect in the\n"
             "response direction, replicated in scRNA", c, weight="bold", fill=0.3, panel=k)
    for u, l in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        down(ax, u, l)


REP_LABEL = {"KPMP": "KPMP sections"}


def panel_e(ax, w, h, N):
    k, c = "e", ACCENT["e"]
    rep = N["cfg"]["shared_program"]["replicate"]
    cohorts = [REP_LABEL.get(x, fl.unit_label(x)) for x in rep["cohorts"]]
    top = header(ax, k, w, h)
    hs = [6.0, 13.4, 8, 8, 8]
    ys = stack(top, hs)
    b1 = box(ax, "e1", 1, ys[0], w - 2, hs[0], f"{len(cohorts)} cohorts never used to define the response genes", c,
             fs=FS * 0.9, weight="bold", fill=0.3, panel=k)
    cw, ch = (w - 2 - 2 * 1.6) / 3, 5.6
    chips = []
    for j, name in enumerate(cohorts):
        row, col = divmod(j, 3)
        x = 1 + col * (cw + 1.6)
        y = ys[1] + hs[1] - (row + 1) * ch - row * 2.2
        chips.append(box(ax, f"e_chip{j}", x, y, cw, ch, name, c, fs=5.9, fill=0.08, panel=k))
    b3 = box(ax, "e3", 1, ys[2], w - 2, hs[2], "Injury–repair score: disease vs healthy AUROC\n"
             f"vs {rep['n_random_sets']:,} random gene sets (matched sizes)", c, panel=k)
    b4 = box(ax, "e4", 1, ys[3], w - 2, hs[3], "Clinical: eGFR, proteinuria, DKD stage,\n"
             "IgAN grade; KPMP donor-level eGFR", c, panel=k)
    b5 = box(ax, "e5", 1, ys[4], w - 2, hs[4], "Sensitivity: without procurement-sensitive\n"
             "or immediate-early genes", c, panel=k)
    down(ax, b1, (1, ys[1], w - 2, hs[1]))
    down(ax, (1, chips[3][1], w - 2, 0), b3)
    down(ax, b3, b4)
    down(ax, b4, b5)


def panel_f(ax, w, h, N):
    k, c = "f", ACCENT["f"]
    top = header(ax, k, w, h)
    hs = [10, 10, 10, 10]
    ys = stack(top, hs)
    b1 = box(ax, "f1", 1, ys[0], w - 2, hs[0], "Genes regressed on a score of the response\n"
             "within patients; each diagnosis\nvs other patients", c, panel=k)
    b2 = box(ax, "f2", 1, ys[1], w - 2, hs[1], "Cell-type marker enrichment patterns\n"
             "replicated in ≥ 2 non-overlapping cohorts", c, panel=k)
    b3 = box(ax, "f3", 1, ys[2], w - 2, hs[2], "Adjusted signatures (50 up, 50 down genes)\n"
             "tested within patients, independent cohorts", c, panel=k)
    b4 = box(ax, "f4", 1, ys[3], w - 2, hs[3], "DKD: loss of podocyte markers (z " + f"{N['podo_z']:.1f}".replace("-", "−") + ")\n"
             + "replicated adjusted signature: " + ", ".join(N["rep_dx"]) + " only", c, weight="bold", fill=0.3,
             panel=k)
    for u, l in ((b1, b2), (b2, b3), (b3, b4)):
        down(ax, u, l)


def build():
    RECORDS.clear()
    N = load_numbers()
    W, H = 170.0, 145.0
    pw, ph, gx, top_pad, gy = 55.0, 66.0, 2.5, 6.0, 6.0
    fig = fl.new_fig(W * fl.MM, H * fl.MM, layout=None)
    axes = {}
    for j, k in enumerate("abcdef"):
        row, col = divmod(j, 3)
        x = col * (pw + gx)
        y = H - top_pad - (row + 1) * ph - row * gy
        axes[k] = panel_axes(fig, (x, y, pw, ph), W, H)
    for k, fn in zip("abcdef", (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f)):
        fn(axes[k], pw, ph, N)
    for k, rec in RECORDS.items():
        fl.save_source(NAME, k, pd.DataFrame(rec))
    return fl.finalize(fig, axes)


if __name__ == "__main__":
    fl.run(build, NAME)
