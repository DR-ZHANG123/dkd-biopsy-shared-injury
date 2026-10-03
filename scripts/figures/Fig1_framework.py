"""Fig. 1 - study framework (panels A-F).

Input (read only): figures/source_data/Fig1_framework_hybrid.svg, the designed framework (editable text and shapes,
embedded organ illustrations). Text corrections are applied here so that the figure matches the manuscript;
the SVG itself is not edited by hand. Output: figures/Fig1.svg, Fig1.pdf (vector) and Fig1.png (300 dpi).
Numbers in the figure: unique specimens, response genes, KPMP DKD decomposition and Shapley shares, replication
AUROCs, DKD eGFR and stage, 177 signatures (all as in the text; checked against the result tables in the text).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "figures" / "source_data" / "Fig1_framework_hybrid.svg"
OUT = ROOT / "figures"
WIDTH_MM = 170
CROP_TOP, CROP_BOTTOM = 96, 978   # SVG units: below the title band, below the bottom panel frames

TEXT = [  # (old text, new text) of whole <text> elements
    ("across 9 cohorts", "from 19 GEO series"),
    ("Other", "Interaction"),
    ("Benchmark", "Tested"),
    ("KPMP donors analyzed in", "KPMP donors analysed in"),
]


def build() -> str:
    s = SRC.read_text()
    for old, new in TEXT:
        n = len(re.findall(rf">{re.escape(old)}</text>", s))
        if n != 1:
            raise SystemExit(f"'{old}': {n} matches")
        s = s.replace(f">{old}</text>", f">{new}</text>")
    # "177" and "published DKD" follow the first word of their line: move them by the width difference
    # (Liberation Sans 18 pt, x-scale 0.9: "Benchmark " 86.4, "Tested " 54.0 SVG units)
    dx = 86.43 - 54.00
    for old_x, tail in (("1221.93", ">177</text>"), ("1256.96", ">published DKD</text>")):
        pat = rf'translate\({old_x},713\)([^>]*{re.escape(tail)})'
        s, n = re.subn(pat, lambda m: f"translate({float(old_x) - dx:.2f},713)" + m.group(1), s)
        if n != 1:
            raise SystemExit(f"{tail}: {n} matches")
    for k in "abcdef":  # panel letters in upper case, as in the captions
        s, n = re.subn(rf'(font-size="31" font-weight="700" fill="#[0-9a-f]{{6}}">){k}</text>', rf"\g<1>{k.upper()}</text>", s)
        if n != 1:
            raise SystemExit(f"panel letter {k}: {n} matches")
    # crop to the six panels: the figure title is given by the legend, not inside the figure
    s = s.replace('viewBox="0 0 1448 1086"', 'viewBox="0 %g 1448 %g"' % (CROP_TOP, CROP_BOTTOM - CROP_TOP), 1)
    s = s.replace('width="1448" height="1086"', 'width="1448" height="%g"' % (CROP_BOTTOM - CROP_TOP), 1)
    return s


def main() -> None:
    s = build()
    (OUT / "Fig1.svg").write_text(s)
    w, h = (float(x) for x in re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', s).groups())
    px = round(WIDTH_MM / 25.4 * 300)
    cairosvg.svg2pdf(bytestring=s.encode(), write_to=str(OUT / "Fig1.pdf"),
                     output_width=WIDTH_MM / 25.4 * 72, output_height=WIDTH_MM / 25.4 * 72 * h / w)
    # the embedded illustrations are ~6,000 ppi; downsample images to 600 ppi (text and shapes stay vector)
    raw = OUT / "Fig1_raw.pdf"
    (OUT / "Fig1.pdf").replace(raw)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.6",
                    "-dDownsampleColorImages=true", "-dColorImageDownsampleType=/Bicubic", "-dColorImageResolution=600",
                    "-dDownsampleGrayImages=true", "-dGrayImageDownsampleType=/Bicubic", "-dGrayImageResolution=600",
                    "-dEmbedAllFonts=true", "-dDetectDuplicateImages=true", f"-sOutputFile={OUT / 'Fig1.pdf'}", str(raw)],
                   check=True)
    raw.unlink()
    cairosvg.svg2png(bytestring=s.encode(), write_to=str(OUT / "Fig1.png"), output_width=px, output_height=round(px * h / w))
    print(f"wrote Fig1: {WIDTH_MM} x {WIDTH_MM * h / w:.0f} mm -> figures/Fig1.svg, Fig1.pdf, Fig1.png")


if __name__ == "__main__":
    sys.exit(main())
