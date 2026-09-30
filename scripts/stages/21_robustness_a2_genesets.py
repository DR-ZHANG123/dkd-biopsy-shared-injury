"""Stage 21 A2（输入准备）：把 results/21_robustness/A2_transplant/genesets/raw 中的 ATAGC（Halloran/Famulski）移植肾基因集
原始文件解析为逐集 TSV（下载 URL 与 md5 见 genesets/sources.tsv）。
Parse ATAGC (Halloran/Famulski) transplant gene sets from the raw files in genesets/raw into per-set TSVs.

No gene is added or edited by hand: every row comes from a raw file listed in sources.tsv.
Inputs
  raw/jasn2012_sdc1.pdf                     JASN 2012 Supplemental Table 1 (394 IRRATs, FC, P)
  raw/converted_xlsx/*.xlsx                 LibreOffice xls->xlsx conversions of raw/atagc_genelists_wayback/*.xls
  raw/atagc_genelists_wayback/cIRIT_HT.xlsx native xlsx
Run:  conda activate kir; python scripts/stages/21_robustness_a2_genesets.py
Re-create the .xlsx conversions (pandas here has no xlrd):
  libreoffice --headless --convert-to xlsx --outdir raw/converted_xlsx raw/atagc_genelists_wayback/*.xls
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2] / "results" / "21_robustness" / "A2_transplant" / "genesets"
RAW = HERE / "raw"

# sheet -> (output set name, direction as defined by the source)
CORE = {
    "IRRAT": ("IRRAT30", "up (early AKI biopsies vs 6-wk pristine protocol biopsies)"),
    "IRITD3": ("IRITD3", "up (induced by non-immune injury in mouse isografts, peak day 3)"),
    "IRITD5": ("IRITD5", "up (induced by non-immune injury in mouse isografts, peak day 5)"),
    "KT1": ("KT1", "down (kidney-selective; majority decreased after injury)"),
    "KT2": ("KT2", "down (epithelial solute carriers reduced after injury or rejection)"),
    "KT1.1": ("KT1.1", "down (kidney-selective, reduced >90% in day-21 mouse allografts)"),
    "KT2.1": ("KT2.1", "down (solute carriers reduced >90% in day-21 mouse allografts)"),
    "ENDAT": ("ENDAT", "cell_type_associated (endothelial; literature-based, not a DE set)"),
    "QCAT": ("QCAT", "cell_type_associated (CD8+ CTL)"),
    "QCMAT": ("QCMAT", "cell_type_associated (constitutive macrophage)"),
    "AMAT1": ("AMAT1", "cell_type_associated (alternatively activated macrophage)"),
    "BAT": ("BAT", "cell_type_associated (B cell)"),
    "IGT": ("IGT", "cell_type_associated (immunoglobulin transcripts)"),
    "NKB": ("NKB", "cell_type_associated (NK cell burden)"),
    "TCB": ("TCB", "cell_type_associated (effector T cell burden)"),
    "GRIT1": ("GRIT1", "up (IFNG- and rejection-induced)"),
    "DSAST": ("DSAST", "up (DSA+ vs DSA- biopsies)"),
    "MCAT": ("MCAT", "up (correlated with scarring; sheet labelled Mast cell associated transcripts)"),
}
MOUSE = {  # file stem -> (set name, direction)
    "mIRITs": ("mouse_IRIT", "up (induced in mouse kidney isografts vs normal kidney)"),
    "mRTs": ("mouse_RT", "down (renal transcripts; high in normal kidney, lost with injury/rejection)"),
    "mSlcs": ("mouse_SLC", "down (renal solute carriers; high in normal kidney, lost with injury/rejection)"),
    "mCATs": ("mouse_CAT", "up (CTL-associated, increased in rejection)"),
    "mtGRITs": ("mouse_tGRIT", "up (true IFNG- and rejection-induced)"),
    "moGRITs": ("mouse_oGRIT", "up (occult IFNG- and rejection-induced)"),
    "mGSTs": ("mouse_GST", "up (IFNG-suppressed; over-expressed without IFNG)"),
    "mAMATs": ("mouse_AMAT", "up (alternative macrophage activation)"),
    "mCISTs": ("mouse_CIST", "up (class I-suppressed; over-expressed in class I-deficient allografts)"),
}
HEART = {"cIRIT": ("heart_cIRIT3", "up (cardiac isograft injury-repair)"),
         "HTs": ("heart_HT", "down (heart-selective transcripts)")}


def split_symbols(raw, title=""):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if hasattr(raw, "month"):  # Excel-mangled symbol stored as a date in the source file
        if raw.month == 9 and str(title).lower().startswith("septin"):
            return [f"Sept{raw.day}"]
        raise ValueError(f"unhandled date-like symbol {raw!r} ({title})")
    s = str(raw).strip()
    if s in ("", "---", "nan"):
        return []
    if s == "miR21/VMP1":  # compound label in Core PBTs IRRAT sheet (probe 224917_at)
        return ["VMP1", "MIR21"]
    out = []
    for t in s.split("///"):
        t = t.strip().split(";")[0].strip()  # "EDN1; ET1" -> primary symbol, aliases kept in symbol_raw
        if t and t != "---":
            out.append(t)
    return out


def parse_pbt_sheet(df):
    """Return (metadata dict, data frame with probe_set/gene_title/symbol_raw)."""
    hdr = next(i for i in range(len(df)) if re.search("probe", str(df.iat[i, 0]), re.I))
    meta = {str(df.iat[i, 0]).strip().rstrip(":").strip(): str(df.iat[i, 1]).strip()
            for i in range(hdr) if pd.notna(df.iat[i, 0])}
    cols = [str(c).strip() for c in df.iloc[hdr]]
    body = df.iloc[hdr + 1:].copy()
    body.columns = cols
    body = body[body.iloc[:, 0].notna()]
    sym_col = next(c for c in cols if "symbol" in c.lower())
    title_col = next(c for c in cols if "gene title" in c.lower() or "gene name" in c.lower())
    out = pd.DataFrame({"probe_set": body.iloc[:, 0].astype(str).str.strip(),
                        "gene_title": body[title_col].astype(str),
                        "symbol_raw": body[sym_col]})
    out["file_row"] = range(1, len(out) + 1)
    return meta, out


def explode(df, direction, stat_fn):
    rows = []
    for r in df.itertuples(index=False):
        for s in split_symbols(r.symbol_raw, r.gene_title):
            raw = r.symbol_raw
            if hasattr(raw, "month"):
                raw = f"EXCEL_DATE:{raw.date()} (restored from gene title)"
            rows.append({"gene_symbol": s, "probe_set": r.probe_set, "direction": direction,
                         "source_rank_or_stat": stat_fn(r), "symbol_raw": raw,
                         "gene_title": r.gene_title})
    return pd.DataFrame(rows, columns=["gene_symbol", "probe_set", "direction",
                                       "source_rank_or_stat", "symbol_raw", "gene_title"])


def parse_irrat394():
    txt = subprocess.run(["pdftotext", "-layout", str(RAW / "jasn2012_sdc1.pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    pat = re.compile(r"^\f?(\S+_at)\s+([\d.]+)\s+([\d.Ee-]+)\s+(\S+)\s*(.*)$")
    rows = []
    for line in txt.splitlines():
        m = pat.match(line)
        if not m:
            continue
        probe, fc, p, sym, rest = m.groups()
        if sym == "VMP1/":  # "VMP1/ mir-21likely ortholog ..." (symbol column overflows)
            sym, rest = "VMP1/ mir-21", rest[len("mir-21"):]
        rows.append({"probe_set": probe, "fc": float(fc), "p": p, "symbol_raw": sym,
                     "gene_title": rest.strip()})
    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)
    return df


def irrat394_symbols(sym):
    if sym == "VMP1/ mir-21":
        return ["VMP1", "MIR21"]
    return [sym]


def main():
    summary = []
    # 1) IRRAT394 from JASN Supplemental Table 1
    irr = parse_irrat394()
    rows = []
    for r in irr.itertuples(index=False):
        for s in irrat394_symbols(r.symbol_raw):
            rows.append({"gene_symbol": s, "probe_set": r.probe_set,
                         "direction": "up (early AKI biopsies vs 6-wk pristine protocol biopsies)",
                         "source_rank_or_stat": f"rank={r.rank};FC_AKI_vs_control={r.fc};P={r.p}",
                         "symbol_raw": r.symbol_raw, "gene_title": r.gene_title})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "IRRAT394.tsv", sep="\t", index=False)
    summary.append(("IRRAT394", len(irr), out.gene_symbol.nunique()))
    fc_by_probe = {r.probe_set: (r.rank, r.fc, r.p) for r in irr.itertuples(index=False)}

    # 2) ATAGC Core PBTs (human)
    core = pd.read_excel(RAW / "converted_xlsx/Core_PBTs.xlsx", sheet_name=None, header=None)
    for sheet, (name, direction) in CORE.items():
        meta, df = parse_pbt_sheet(core[sheet])
        if name == "IRRAT30":
            def stat(r):
                if r.probe_set in fc_by_probe:
                    k, fc, p = fc_by_probe[r.probe_set]
                    return f"JASN2012_SuppT1_rank={k};FC_AKI_vs_control={fc};P={p}"
                return f"file_row={r.file_row};probe_not_in_JASN_SuppT1"
        else:
            def stat(r):
                return f"file_row={r.file_row}"
        out = explode(df, direction, stat)
        out.to_csv(HERE / f"{name}.tsv", sep="\t", index=False)
        summary.append((name, len(df), out.gene_symbol.nunique()))
        if name == "IRRAT30":  # AKI signal of Famulski AJT 2013 = top-30 IRRATs (methods text)
            ak = out.copy()
            ak.to_csv(HERE / "AKIT30.tsv", sep="\t", index=False)
            summary.append(("AKIT30", len(df), ak.gene_symbol.nunique()))

    # 3) ATAGC mouse PBTs and human-heart PBTs
    for stem, (name, direction) in MOUSE.items():
        wb = pd.read_excel(RAW / f"converted_xlsx/{stem}.xlsx", sheet_name=None, header=None)
        sheet = next(k for k, v in wb.items() if not v.empty)
        meta, df = parse_pbt_sheet(wb[sheet])
        out = explode(df, direction, lambda r: f"file_row={r.file_row}")
        out.to_csv(HERE / f"{name}.tsv", sep="\t", index=False)
        summary.append((name, len(df), out.gene_symbol.nunique()))
    wb = pd.read_excel(RAW / "atagc_genelists_wayback/cIRIT_HT.xlsx", sheet_name=None, header=None)
    for sheet, (name, direction) in HEART.items():
        meta, df = parse_pbt_sheet(wb[sheet])
        out = explode(df, direction, lambda r: f"file_row={r.file_row}")
        out.to_csv(HERE / f"{name}.tsv", sep="\t", index=False)
        summary.append((name, len(df), out.gene_symbol.nunique()))

    s = pd.DataFrame(summary, columns=["set", "n_rows_in_source_file", "n_unique_symbols_parsed"])
    s.to_csv(HERE / "_parse_summary.tsv", sep="\t", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
