"""Stage 17a：独立测试集候选的样本构成与查重（不训练、不打分、不看任何模型输出）。

- 候选：KPMP 区域转录组（LMD：GLOM / TI）与活检切片 bulk（开放获取层）、GSE162830（LMD 肾小球）、
  GSE166239（FFPE 全切片）、GSE45980（冷冻切片 Agilent）。读取与标签映射见 lib/indep_candidates.py。
- 样本构成：按 (候选, 区室, 诊断) 计数（参与者层面），并与冻结协议的最小样本量（model2.eval.min_test_pos/neg）比较。
- 查重：沿用 stage 02 的表达残差相关（同一套 rank_z / compare 与阈值 config.overlap），
  每个候选 × 现有全部 data/interim series（含候选自身、候选之间）；
  阳性对照 = KPMP 19-0001 同一文库在 HiSeq 与 NovaSeq 上的两次测序（应被判为重复）。
产出：results/17_independent/candidates/{candidate_samples,composition,feasibility,overlap_pairs,overlap_summary}.tsv
      + PROVENANCE.json；候选表达矩阵 data/interim/indep/<cohort>_expr.parquet。
"""
from __future__ import annotations

import itertools
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.geo import GeneIndex  # noqa: E402
from lib.indep_candidates import load_gse45980, load_gse162830, load_gse166239, load_kpmp  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OV = CFG["overlap"]
EV = CFG["model2"]["eval"]
INTERIM = ROOT / CFG["paths"]["interim"]
OUT = ROOT / "results" / "17_independent" / "candidates"
RAW = ROOT / CFG["paths"]["raw"]
GENE_INFO = (ROOT / CFG["old_project_raw"]).resolve() / "metadata" / "Homo_sapiens.gene_info.gz"
OTHER_CKD = set(CFG["tasks"]["T2"]["neg"]) | {"OTHER", "ING", "OTHER_CKD"}   # 与 stage 15 阴性类同义；ING 为非糖尿病结节性硬化


def load_all() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    gidx = GeneIndex(GENE_INFO)
    parts = {"KPMP": load_kpmp(RAW / "kpmp_lmd", gidx), "GSE162830": load_gse162830(RAW / "indep", gidx),
             "GSE166239": load_gse166239(RAW / "indep", gidx, GENE_INFO), "GSE45980": load_gse45980(RAW / "indep", gidx)}
    (INTERIM / "indep").mkdir(parents=True, exist_ok=True)
    for k, (e, s) in parts.items():
        e.astype("float32").to_parquet(INTERIM / "indep" / f"{k}_expr.parquet")
    S = pd.concat([s for _, s in parts.values()])
    return {k: e for k, (e, _) in parts.items()}, S


def composition(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """参与者层面：每个 (候选, 区室) 每类计一次（同一参与者同区室多个文库只计一次）。"""
    x = S[S.compartment.isin(["GLOM", "TUB", "WHOLE"])].drop_duplicates(["cohort", "participant", "compartment"])
    comp = x.groupby(["cohort", "compartment", "diagnosis"]).participant.nunique().rename("n_participants").reset_index()
    rows = []
    for (c, k), g in x.groupby(["cohort", "compartment"]):
        n = g.diagnosis.value_counts()
        dkd, ctrl = int(n.get("DKD", 0)), int(n.get("CONTROL", 0))
        oth = int(n[[d for d in n.index if d in OTHER_CKD]].sum())
        und = int(n.get("UNDETERMINED", 0))
        rows.append({"cohort": c, "compartment": k, "n_dkd": dkd, "n_other_ckd": oth, "n_ckd_undetermined": und,
                     "n_control": ctrl, "n_aki": int(n[[d for d in n.index if d.startswith("AKI")]].sum()),
                     "primary_ok": dkd >= EV["min_test_pos"] and oth >= EV["min_test_neg"],
                     "secondary_ok": dkd >= EV["min_test_pos"] and ctrl >= EV["min_test_neg"]})
    return comp, pd.DataFrame(rows)


def overlap(data_c: dict[str, pd.DataFrame], S: pd.DataFrame) -> pd.DataFrame:
    st2 = import_module("02_overlap_audit")
    ref = {p.name.replace("_expr.parquet", ""): p for p in sorted(INTERIM.glob("*_expr.parquet"))}
    ref_comp = pd.read_csv(ROOT / "results/02_overlap/sample_table.tsv", sep="\t", usecols=["sample_uid", "compartment"]
                           ).set_index("sample_uid").compartment
    out = []
    keys = list(data_c)
    pairs = [(a, b) for a in keys for b in ref] + list(itertools.combinations_with_replacement(keys, 2))
    cache: dict[str, pd.DataFrame] = {}
    for a, b in pairs:
        ea = data_c[a]
        eb = data_c[b] if b in data_c else cache.setdefault(b, pd.read_parquet(ref[b]))
        m, _ = st2.compare(a, ea, b, eb, [])
        for r in m:
            r["within"] = a == b
            r["candidate_compartment"] = S.compartment.get(r["gsm_a"], "NA")
            ub = r["gsm_b"] if b in data_c else f"{b}|{r['gsm_b']}"
            r["other_compartment"] = S.compartment.get(ub, ref_comp.get(ub, "NA"))
        out += m
        print(f"{a} vs {b}: genes={len(ea.index.intersection(eb.index))} mutual={len(m)}", flush=True)
    P = pd.DataFrame(out)
    thr = np.where(P.within, OV["resid_min_within"], OV["resid_min"])
    P["expr_dup"] = P.resid_r >= thr
    return P


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    data_c, S = load_all()
    S.to_csv(OUT / "candidate_samples.tsv", sep="\t")
    comp, feas = composition(S)
    comp.to_csv(OUT / "composition.tsv", sep="\t", index=False)
    feas.to_csv(OUT / "feasibility.tsv", sep="\t", index=False)
    print(feas.to_string())
    P = overlap(data_c, S)
    P.to_csv(OUT / "overlap_pairs.tsv", sep="\t", index=False)
    summ = P.groupby(["series_a", "series_b"]).agg(n_mutual=("resid_r", "size"), max_resid=("resid_r", "max"),
                                                    q99_resid=("resid_r", lambda v: float(np.quantile(v, 0.99))),
                                                    n_expr_dup=("expr_dup", "sum")).reset_index()
    summ.to_csv(OUT / "overlap_summary.tsv", sep="\t", index=False)
    print(summ[summ.n_expr_dup > 0].to_string())
    ins = [RAW / "kpmp_lmd" / "checksums.tsv", ROOT / "results/00_download/indep_candidates_checksums.tsv"]
    outs = [OUT / f for f in ("candidate_samples.tsv", "composition.tsv", "feasibility.tsv", "overlap_pairs.tsv",
                              "overlap_summary.tsv")]
    write_provenance("17_independent/candidates", ins + sorted(INTERIM.glob("*_expr.parquet")), outs, CFG["seed"],
                     {"min_test_pos": EV["min_test_pos"], "min_test_neg": EV["min_test_neg"],
                      "feasibility": feas.to_dict("records")})


if __name__ == "__main__":
    main()
