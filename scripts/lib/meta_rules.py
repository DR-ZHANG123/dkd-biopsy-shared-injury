"""GEO 样本注释 → 区室 / 研究方标本编号 / 诊断。规则均按各 series 的真实文本写成（stage 02、03 共用）。

标本编号（biopsy_id）只用于查重，不含诊断信息以外的推断；编码表见 ERCB_CODE / NEPTUNE_CODE。
"""
from __future__ import annotations

import re

import pandas as pd

# ---------- 区室 ----------
TUB_PAT = re.compile(r"tubul", re.I)
GLOM_PAT = re.compile(r"glomerul(i|us|ar compartment)\b|\bglom\b", re.I)
WHOLE_PAT = re.compile(r"biops|kidney|renal", re.I)


def _fields(row: pd.Series, prefixes: tuple[str, ...]) -> list[str]:
    return [str(v) for k, v in row.items() if k.startswith(prefixes) and isinstance(v, str) and v]


def tissue_text(row: pd.Series) -> str:
    """优先用 'tissue:' 特征字段；没有时再退到 source_name / 旧版 compartment 字段。
    不用标题与描述：其中的病名（如 Glomerulosclerosis）会误导区室判断。"""
    tis = [v for v in _fields(row, ("characteristics",)) if v.lower().startswith(("tissue", "tissue subregion"))]
    if "compartment" in row and isinstance(row["compartment"], str):
        tis.append(row["compartment"])
    if not tis:
        tis = _fields(row, ("source_name",))
    return " ; ".join(tis)


def classify_compartment(row: pd.Series) -> str:
    t = tissue_text(row)
    # "tissue subregion: tubules" / "glomerulus" 比 "tissue: kidney" 更具体，先判区室
    if TUB_PAT.search(t):
        return "TUB"
    if GLOM_PAT.search(t):
        return "GLOM"
    if WHOLE_PAT.search(t):
        return "WHOLE"
    return "UNKNOWN"


# ---------- 研究方标本编号 ----------
# ERCB 标题：H7-Glom-DN901、H7-Glom-LD_974、H7-Glom-FSGS&MCD905、H5-Tub-SLE465；
# 例外：GSE47183 的 'H1-Glom-TMD' 无编号（只能靠表达查重）。
# GSE32591（ERCB LN 研究）：Tub_LN_301、Glom_LD_480。编号在「区室 × 病种代码」内唯一，批次前缀不参与匹配。
ERCB_PAT = re.compile(r"\bH(\d+)-(Glom|Tub)-([A-Za-z&]+?)_?(\d+)(?:_\d+)?\b")   # "TN997_1" = TN997 的重复杂交
ERCB_LN_PAT = re.compile(r"^(Glom|Tub)_(LN|LD)_(\d+)$")
ERCB_CODE = {"DN": "DN", "LD": "LD", "TN": "TN", "MCD": "MCD", "FSGS": "FSGS", "FSGS&MCD": "FSGS&MCD",
             "MGN": "MGN", "SLE": "SLE", "LN": "SLE", "IGA": "IGA", "HT": "HT", "RPGN": "RPGN", "TMD": "TMD"}
# NEPTUNE：GSE108109 "[B1_Glom_FSGS_1]"、GSE108112 "[B1_TI_FSGS_1]"、GSE133288 "B1_TI_MN_94"、
# GSE200818 "94_MN"。编号在病种代码内唯一（FSGS_2 与 LivingDonor_2 是不同患者），批次号不参与匹配。
NEPTUNE_PAT = re.compile(r"\bB\d+_(Glom|TI)_([A-Za-z]+)_(\d+)\b")
NEPTUNE_200818_PAT = re.compile(r"^(\d+)_([A-Za-z]+)$")
NEPTUNE_CODE = {"LivingDonor": "LD", "LD": "LD", "FSGS": "FSGS", "MCD": "MCD", "MN": "MN",
                "Other": "Other", "AAV": "AAV"}
ILLUMINA_BARCODE = re.compile(r"^\d{10}_[A-L]$")          # WG-DASL 芯片条码（!Sample_description）


def biopsy_id(gse: str, row: pd.Series, compartment: str) -> str | None:
    title = str(row.get("title", ""))
    comp = {"Glom": "GLOM", "Tub": "TUB", "TI": "TUB"}
    m = ERCB_PAT.search(title)
    if m:
        return f"ERCB:{comp[m[2]]}:{ERCB_CODE[m[3].upper()]}:{int(m[4])}"
    m = ERCB_LN_PAT.match(title)
    if m and gse == "GSE32591":
        return f"ERCB:{comp[m[1]]}:{ERCB_CODE[m[2]]}:{int(m[3])}"
    m = NEPTUNE_PAT.search(title)
    if m:
        return f"NEPTUNE:{comp[m[1]]}:{NEPTUNE_CODE[m[2]]}:{int(m[3])}"
    m = NEPTUNE_200818_PAT.match(title)
    if m and gse == "GSE200818":
        return f"NEPTUNE:{compartment}:{NEPTUNE_CODE[m[2]]}:{int(m[1])}"
    desc = str(row.get("description", ""))
    if ILLUMINA_BARCODE.match(desc):
        return f"ILMN:{desc}"
    return None


# ---------- 诊断 ----------
ERCB_DIAG = {"DN": "DKD", "LD": "CONTROL", "TN": "CONTROL", "MCD": "MCD", "FSGS": "FSGS",
             "FSGS&MCD": "FSGS_MCD", "MGN": "MN", "SLE": "LN", "IGA": "IgAN", "HT": "HT",
             "RPGN": "RPGN", "TMD": "TMD"}
NEPTUNE_DIAG = {"LD": "CONTROL", "FSGS": "FSGS", "MCD": "MCD", "MN": "MN", "Other": "OTHER", "AAV": "RPGN"}
# 自由文本规则：顺序即优先级（更具体者在前）。只作用于诊断相关字段，不作用于标题中的编号。
DIAG_RULES = [
    ("DKD", r"diabetic (nephropathy|kidney)|\bDKD\b|^DN$|Early_DN|Advanced_DN"),
    ("OTHER", r"non[- ]IgAN|glomerular disease \(other\)|other glomerular disease|other nephrotic"
              r"|nephrotic syndrome, unspecified|\bCKD\b"),
    ("LN", r"lupus|\bLN patient\b|\bSLE\b"),
    ("IgAN", r"\bIgA|IgAN"),
    ("FSGS_MCD", r"FSGS[/ &_-]*MCD|glomerul(ar )?sclerosis ?(/|and) ?minimal change"),
    ("FSGS", r"focal (and )?segmental|\bFSGS\b"),
    ("MCD", r"minimal change"),
    ("MN", r"membranous"),
    ("HT", r"hypertensi|nephrosclerosis"),
    ("RPGN", r"rapidly progressive|\bRPGN\b|vasculitis|\bANCA\b"),
    ("TMD", r"thin (basement )?membran"),
    ("CONTROL", r"living donor|tumou?r nephrectom|healthy|\bnormal\b|\bcontrol\b|^LD$|\bLD\b"),
]
DIAG_KEYS = ("diagnosis", "disease", "disease state", "disease status", "subject status", "group")


def diagnosis_text(row: pd.Series) -> str:
    """characteristics 中 key 属于 DIAG_KEYS 的值 + source_name（病名常写在这里）+ 旧版 group 字段。"""
    vals = []
    for v in _fields(row, ("characteristics",)):
        k, _, val = v.partition(":")
        if k.strip().lower() in DIAG_KEYS and val.strip():
            vals.append(val.strip())
    vals += _fields(row, ("source_name",))
    if "group" in row and isinstance(row["group"], str):
        vals.insert(0, row["group"])
    return " ; ".join(vals)


def classify_text(text: str) -> str:
    for lab, pat in DIAG_RULES:
        if any(re.search(pat, part.strip(), flags=re.I) for part in text.split(";")):
            return lab
    return "UNKNOWN"


def classify_diagnosis(row: pd.Series, bid: str | None) -> tuple[str, str]:
    """返回 (diagnosis, 依据)。标本编号里的病种代码最稳定，优先使用。"""
    if bid and bid.startswith("ERCB:"):
        return ERCB_DIAG[bid.split(":")[2]], "ERCB title code"
    if bid and bid.startswith("NEPTUNE:"):
        return NEPTUNE_DIAG[bid.split(":")[2]], "NEPTUNE title code"
    t = diagnosis_text(row)
    return classify_text(t), f"text: {t[:120]}"
