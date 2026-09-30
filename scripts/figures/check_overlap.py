"""Measured layout check of every figure: renders each figure with matplotlib and inspects text bboxes.

Checks per figure (a figure passes only when every count is 0):
  text_text     pairs of drawn text artists whose bboxes intersect (> 0.5 px each way at 300 dpi)
  text_axes     text that belongs to one axes (or to the figure) intruding into the data area of another axes
  out_axes      text/legend inside an axes that extends beyond that axes (unless gid 'free')
  out_canvas    text outside the figure canvas
  out_box       text with gid 'in:<id>' not fully inside patch 'box:<id>' (schematics)
  axes_axes     overlapping axes (twins and gid 'inset' excluded)
  tick_gap      adjacent drawn tick labels of one axis closer than 1 pt (touching labels read as one, e.g. "0.40.8")
  tight_lines   multi-line text drawn with line spacing < 1.0 (lines of one Text object crowd each other)
  label_align   panel letters of one row (or column) not on one line (> 0.5 px)
  size          canvas wider than 170 mm or taller than 225 mm (graphical abstract: 920 x 300 px)
Usage: python scripts/figures/check_overlap.py [Fig2 FigS1 ...]   (default: all figures)
Writes figures/check_overlap_report.tsv and prints a summary.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figlib  # noqa: E402

FIGURES = ["Fig1_framework", "Fig2_shared_program", "Fig3_cells", "Fig4_repair_state", "Fig5_replication",
           "Fig6_residuals", "FigS1_scale_audit", "FigS2_resource", "FigS3_split_controls",
           "FigS4_egfr", "FigS5_core_stability", "FigS6_deconv", "FigS7_kpmp_decomposition", "FigS8_repair_bulk",
           "FigS9_kpmp_markers", "FigS10_sc_direction", "FigS11_model_eval",
           "FigS12_signatures",
           "FigS13_stratified", "FigS14_injection", "graphical_abstract"]
TOL = 0.5  # px


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drawn_texts(fig, r):
    """All visible, non-empty text artists as drawn, with their owner axes (None for figure-level)."""
    tick_all, tick_drawn = set(), set()
    owner = {}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_all.update({id(t.label1), id(t.label2)})
            if not ax.axison:  # axis switched off: tick labels are not drawn
                continue
            for t in axis._update_ticks():
                for lab in (t.label1, t.label2):
                    if lab.get_visible():
                        tick_drawn.add(id(lab))
        for t in ax.findobj(Text):
            owner.setdefault(id(t), ax)
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in tick_all and id(t) not in tick_drawn:
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        bb = t.get_window_extent(r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((t, bb, owner.get(id(t))))
    return out


def inter(a, b, tol=TOL) -> bool:
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def inside(a, b, tol=TOL) -> bool:
    return a.x0 >= b.x0 - tol and a.x1 <= b.x1 + tol and a.y0 >= b.y0 - tol and a.y1 <= b.y1 + tol


def in_legend(t, fig) -> Legend | None:
    for leg in fig.findobj(Legend):
        if t in leg.get_texts() or t is leg.get_title():
            return leg
    return None


def check(fig, name: str) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = drawn_texts(fig, r)
    res = {k: [] for k in ("text_text", "text_axes", "out_axes", "out_canvas", "out_box", "axes_axes",
                           "tick_gap", "tight_lines", "label_align", "size")}
    lab = lambda t: repr(t.get_text()[:30])  # noqa: E731
    # text-text
    for (t1, b1, _), (t2, b2, _) in itertools.combinations(texts, 2):
        if inter(b1, b2):
            res["text_text"].append(f"{lab(t1)} x {lab(t2)}")
    # data areas of visible axes (schematic axes with axis off still count: their texts are children)
    areas = [(ax, ax.patch.get_window_extent(r)) for ax in fig.axes if ax.get_visible() and ax.axison]
    for t, bb, own in texts:
        if t.get_gid() == "free_any":
            continue
        for ax, ab in areas:
            if ax is own or (own is not None and ax.bbox.bounds == own.bbox.bounds):
                continue
            if inter(bb, ab, tol=1.0):
                res["text_axes"].append(f"{lab(t)} into axes {ax.get_label() or id(ax)}")
    # out of axes (text children of an axes, legends inside axes)
    for t, bb, own in texts:
        if own is None or t.get_gid() in ("free", "free_any") or not own.axison:
            continue
        if t in (own.title, own._left_title, own._right_title, own.xaxis.label, own.yaxis.label):
            continue
        if own.xaxis.get_offset_text() is t or own.yaxis.get_offset_text() is t:
            continue
        if any(t is x.label1 or t is x.label2 for axis in (own.xaxis, own.yaxis)
               for x in axis.get_major_ticks() + axis.get_minor_ticks()):
            continue
        leg = in_legend(t, fig)
        if leg is not None:
            if leg.get_gid() in ("free", "free_any"):
                continue
            bb = leg.get_window_extent(r)
        if not inside(bb, own.patch.get_window_extent(r)):
            res["out_axes"].append(lab(t))
    # canvas
    canvas = matplotlib.transforms.Bbox.from_bounds(0, 0, W, H)
    for t, bb, _ in texts:
        if not inside(bb, canvas):
            res["out_canvas"].append(lab(t))
    # text inside its box
    boxes = {p.get_gid()[4:]: p.get_window_extent(r) for p in fig.findobj(Patch)
             if isinstance(p.get_gid(), str) and p.get_gid().startswith("box:")}
    for t, bb, _ in texts:
        g = t.get_gid()
        if isinstance(g, str) and g.startswith("in:"):
            b = boxes.get(g[3:])
            if b is None or not inside(bb, b, tol=-1.0):
                res["out_box"].append(f"{lab(t)} ({g})")
    # axes-axes
    axs = [ax for ax in fig.axes if ax.get_visible() and ax.get_gid() != "inset"]
    for a1, a2 in itertools.combinations(axs, 2):
        b1, b2 = a1.get_window_extent(r), a2.get_window_extent(r)
        if b1.bounds == b2.bounds:
            continue
        if inter(b1, b2, tol=1.0):
            res["axes_axes"].append(f"{a1.get_label() or id(a1)} x {a2.get_label() or id(a2)}")
    # tick-label spacing: consecutive drawn labels along the axis direction need >= 1 pt of clearance
    gap_px = 1.0 * fig.dpi / 72
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        for axis, horiz in ((ax.xaxis, True), (ax.yaxis, False)):
            for which in ("label1", "label2"):
                labs = [getattr(t, which) for t in axis._update_ticks()]
                bbs = sorted([lb.get_window_extent(r) for lb in labs if lb.get_visible() and lb.get_text().strip()],
                             key=lambda b: b.x0 if horiz else b.y0)
                for b1, b2 in zip(bbs, bbs[1:]):
                    gap = (b2.x0 - b1.x1) if horiz else (b2.y0 - b1.y1)
                    across = min(b1.y1, b2.y1) - max(b1.y0, b2.y0) if horiz else min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                    if gap < gap_px and across > 0:
                        res["tick_gap"].append(f"axes {ax.get_label() or id(ax)} gap={gap / fig.dpi * 72:.2f}pt")
    for t, bb, _ in texts:
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.0:
            res["tight_lines"].append(f"{lab(t)} linespacing={t._linespacing}")
    # panel letters
    panels = [(t, t.get_window_extent(r)) for t in fig.texts if t.get_gid() == "panel"]
    for (t1, b1), (t2, b2) in itertools.combinations(panels, 2):
        dy, dx = abs(b1.y0 - b2.y0), abs(b1.x0 - b2.x0)
        near_row = dy < 20 * 300 / 72  # within 20 pt vertically -> same row
        near_col = dx < 20 * 300 / 72
        if near_row and dy > TOL:
            res["label_align"].append(f"{t1.get_text()}-{t2.get_text()} row dy={dy:.1f}px")
        if near_col and dx > TOL:
            res["label_align"].append(f"{t1.get_text()}-{t2.get_text()} col dx={dx:.1f}px")
    w_in, h_in = fig.get_size_inches()
    if name == "graphical_abstract":
        if abs(w_in * fig.dpi - 920) > 1 or abs(h_in * fig.dpi - 300) > 1:
            res["size"].append(f"{w_in * fig.dpi:.0f}x{h_in * fig.dpi:.0f}px")
    elif w_in > figlib.FULL_W + 1e-6 or h_in > figlib.MAX_H + 1e-6:
        res["size"].append(f"{w_in / figlib.MM:.0f}x{h_in / figlib.MM:.0f}mm")
    return res


def main(names):
    rows = []
    for name in names:
        if not (HERE / f"{name}.py").exists():
            rows.append((name, "missing", "script not found"))
            continue
        fig = load(name).build()
        res = check(fig, name)
        plt.close(fig)
        n = sum(len(v) for v in res.values())
        rows.append((name, "PASS" if n == 0 else "FAIL",
                     "; ".join(f"{k}={len(v)}" for k, v in res.items())))
        for k, v in res.items():
            for item in v[:15]:
                print(f"  {name} {k}: {item}")
    rep = figlib.FIG_DIR / "check_overlap_report.tsv"
    if len(names) == len(FIGURES):
        with open(rep, "w") as fh:
            fh.write("figure\tstatus\tcounts\n")
            for row in rows:
                fh.write("\t".join(row) + "\n")
    for row in rows:
        print("\t".join(row))
    bad = [r for r in rows if r[1] != "PASS"]
    print("ALL PASS" if not bad else f"{len(bad)} figure(s) failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or FIGURES))
