"""Shared plotting helpers for the Human Genomics figures (style, paths, source data, panel labels, saving).

Plotting conventions:
  * figure scripts only READ tables under results/ or manuscript/numbers/ and never recompute a model;
  * every panel's exact plotting input is written to figures/source_data/<Fig>_panel<x>.tsv;
  * each script exposes build() -> matplotlib Figure (already finalised, panel labels placed) so that
    scripts/figures/check_overlap.py can measure the very figure that is saved;
  * width 170 mm (full) or 85 mm (half), height <= 225 mm, PDF (Type 42 fonts, embedded) + PNG 300 dpi.

gid conventions used by check_overlap.py:
  "panel"       panel letter (placed by finalize(); aligned per row/column)
  "free"        text/legend deliberately outside its axes (still must stay on the canvas)
  "in:<id>"     text that must lie inside the patch whose gid is "box:<id>"
  "inset"       axes allowed to overlap another axes
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
SRC_DIR = FIG_DIR / "source_data"
MM = 1 / 25.4
FULL_W = 170 * MM
HALF_W = 85 * MM
MAX_H = 225 * MM

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
           '#CCB974', '#64B5CD']
# one colour per diagnosis, reused in every figure
DX_COLORS = {"CONTROL": "#8C8C8C", "DKD": "#C44E52", "IgAN": "#4C72B0", "FSGS": "#DD8452", "MCD": "#55A868",
             "FSGS_MCD": "#CCB974", "MN": "#8172B3", "LN": "#DA8BC3", "HT": "#937860", "RPGN": "#64B5CD",
             "TMD": "#BBBBBB", "ING": "#E0A96D", "OTHER": "#6A6A6A", "OTHER_CKD": "#6A6A6A", "REF": "#8C8C8C"}
DX_LABEL = {"CONTROL": "Control", "FSGS_MCD": "FSGS/MCD", "TMD": "TBMD", "OTHER": "Other CKD",
            "OTHER_CKD": "Other CKD", "REF": "Reference"}
COMP_COLORS = {"GLOM": "#4C72B0", "TUB": "#DD8452", "WHOLE": "#55A868"}
COMP_LABEL = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium", "WHOLE": "Whole biopsy"}
METHOD_COLORS = {"RRG-ID": "#C44E52", "B-L2": "#4C72B0", "B2-PCA": "#55A868", "B-cPCA": "#8172B3",
                 "B-rankLASSO": "#DD8452", "B-L2-unitcenter": "#64B5CD", "B-injury": "#8C8C8C"}
METHOD_LABEL = {"B-L2": "L2 logistic", "B2-PCA": "PCA logistic", "B-cPCA": "Contrastive PCA", "B-rankLASSO": "Rank LASSO",
                "B-L2-unitcenter": "L2 logistic (cohort-centred)", "B-injury": "External score", "RRG-ID": "Classifier"}
UNIT_LABEL = {"ERCB_GLOM_H1": "ERCB-GLOM-H1", "ERCB_GLOM_H7": "ERCB-GLOM-H7", "ERCB_TUB_H1": "ERCB-TUB-H1",
              "ERCB_TUB_H7": "ERCB-TUB-H7"}

INPUTS: list[Path] = []


def unit_label(u: str) -> str:
    """Human-readable cohort name: ERCB_GLOM_H1 -> ERCB-GLOM-H1, GSE96804-GPL17586 -> GSE96804."""
    if u in UNIT_LABEL:
        return UNIT_LABEL[u]
    if u.startswith("GSE"):
        return u.split("-")[0]
    return u.replace("_", " ")


def dx_label(d: str) -> str:
    return DX_LABEL.get(d, d)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "axes.labelweight": "bold",
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "patch.linewidth": 0.5, "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC", "legend.borderpad": 0.3, "legend.handlelength": 1.2,
        "legend.labelspacing": 0.3, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 300, "savefig.dpi": 300, "mathtext.default": "regular",
    })


setup_style()


def read(rel: str, **kw) -> pd.DataFrame:
    """Read a results table (path relative to the project root) and register it as an input."""
    p = ROOT / rel
    INPUTS.append(p)
    return pd.read_csv(p, sep=kw.pop("sep", "\t"), **kw)


def read_json(rel: str) -> dict:
    import json
    p = ROOT / rel
    INPUTS.append(p)
    return json.loads(p.read_text())


def kv(rel: str) -> dict:
    """key/value table (manuscript/numbers/*.tsv, figdata/key_numbers.tsv) as a dict of strings."""
    d = read(rel)
    kcol, vcol = d.columns[0], d.columns[1]
    return dict(zip(d[kcol].astype(str), d[vcol].astype(str)))


def save_source(fig: str, panel: str, df: pd.DataFrame) -> Path:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    p = SRC_DIR / f"{fig}_panel{panel}.tsv"
    df.to_csv(p, sep="\t", index=False, float_format="%.6g")
    return p


def new_fig(width: float = FULL_W, height: float = 120 * MM, layout: str | None = "constrained"):
    fig = plt.figure(figsize=(width, height), layout=layout)
    if layout == "constrained":
        fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.04, hspace=0.04)
    return fig


def reserve_label(ax, title: str = "", **kw) -> None:
    """Reserve headroom for the panel letter: a (possibly empty) centred title in 11-pt-high line."""
    ax.set_title(title if title else " ", fontsize=kw.pop("fontsize", 7.5 if title else 12),
                 loc=kw.pop("loc", "center"), pad=kw.pop("pad", 3), **kw)


def finalize(fig, labels: dict, pad_pt: float = 2.0, tol: float = 0.03):
    """Freeze the layout and place bold uppercase panel letters (A, B, ...) aligned per row and per column.

    labels: {'a': ax, ...}; keys stay lowercase internally (source-data file names) and are drawn in
    upper case, as in Human Genomics articles (Fig. 2A). Row y = highest axes top (+pad) among axes
    whose tops lie within `tol` (figure fraction); column x = leftmost tight-bbox edge among axes whose left edges lie within `tol`.
    """
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    info = {}
    for k, ax in labels.items():
        pos = ax.get_position()
        tb = ax.get_tightbbox(r, for_layout_only=True)
        # exclude the title from the tight bbox: we want the left edge of ticks/labels
        x0 = min(inv.transform((tb.x0, 0))[0], pos.x0)
        for t in (ax.yaxis.label,):
            if t.get_text().strip() and t.get_visible():
                x0 = min(x0, inv.transform((t.get_window_extent(r).x0, 0))[0])
        info[k] = dict(top=pos.y1, left=pos.x0, x0=x0)
    keys = list(labels)
    ys, xs = {}, {}
    for k in keys:
        row = [j for j in keys if abs(info[j]["top"] - info[k]["top"]) < tol]
        col = [j for j in keys if abs(info[j]["left"] - info[k]["left"]) < tol]
        ys[k] = max(info[j]["top"] for j in row)
        xs[k] = min(info[j]["x0"] for j in col)
    dy = pad_pt / 72 / fig.get_figheight()
    for k in keys:
        fig.text(xs[k], ys[k] + dy, k.upper(), fontsize=11, fontweight="bold", ha="left", va="bottom", gid="panel")
    return fig


def save(fig, name: str, formats=("pdf", "png")) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in formats:
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03, metadata={"CreationDate": None}
                    if ext == "pdf" else None)
        outs.append(p)
    w, h = fig.get_size_inches()
    if w > FULL_W + 1e-6 or h > MAX_H + 1e-6:
        print(f"WARNING {name}: canvas {w / MM:.0f} x {h / MM:.0f} mm exceeds 170 x 225 mm", file=sys.stderr)
    print(f"wrote {name}: {w / MM:.0f} x {h / MM:.0f} mm ->", ", ".join(str(o.relative_to(ROOT)) for o in outs))
    return outs


def run(build, name: str) -> None:
    fig = build()
    save(fig, name)
    plt.close(fig)


# ---------------------------------------------------------------- small plotting helpers
def jitter(n: int, width: float = 0.25, seed: int = 20260928) -> np.ndarray:
    """Deterministic horizontal jitter for strip plots."""
    return np.random.default_rng(seed + n).uniform(-width, width, n)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.2f}" if p >= 0.01 else f"P = {p:.3f}"


def hide(ax) -> None:
    ax.set_axis_off()
