"""Stage 08：RRG 与 B3–B6 的下游评估。

- probe（主分析）：冻结编码器 → 64 维 embedding → 标准化 → L2 logistic（inner CV 选 C，inner OOF Youden 阈值）。
  每个预训练变体一个模型名：RRG-full / B3-noedge / B4-rewired / B6-ppi / B6-coexpr。
- scratch（B5-scratch）：同一架构（full 边）不预训练，直接监督训练；训练 cohort 内分层留出 val_frac 做早停与阈值。
- finetune（RRG-full-ft）：full 预训练权重 + 小学习率全参微调，其余同 scratch。
- label-efficiency：config eval.label_fracs；抽样函数与种子与 stage 06 相同（配对）。
- 种子：全标签下的 probe 只有编码器种子带来的随机性（每个预训练种子一行）；抽样或随机训练时 probe.seeds 个种子。
- --inner-only：只做 inner CV / 内部留出评估，不对测试队列打分（设计冻结前的迭代只能看这个）。

产出：results/08_models/predictions.tsv、thresholds.tsv、inner_cv.tsv、PROVENANCE.json
用法：python 08_probe_finetune.py [--task T2 --compartment GLOM --fold X] [--families probe,scratch] [--device cuda:0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # 共享服务器：限制 BLAS 线程，避免 joblib/torch 超额订阅

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import evalkit as E  # noqa: E402
from lib import nnkit as K  # noqa: E402
from lib.model import build_encoder  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
MODELS = ROOT / CFG["paths"]["models"]


def families() -> list[str]:
    return ["probe", "scratch"] + (["finetune"] if CFG["supervised"]["enable_finetune"] else [])


def enc_path(fold: D.Fold, variant: str, seed: int) -> Path:
    return MODELS / fold.key / variant / f"seed{seed}" / "encoder.pt"


def _inner_rows(base, name, seed, coh, y, oof) -> list[dict]:
    rows = []
    for c in np.unique(coh):
        mk = (coh == c) & ~np.isnan(oof)
        if len(np.unique(y[mk])) == 2:
            rows.append({**base, "model": name, "seed": seed, "inner_cohort": c, "n": int(mk.sum()),
                         "auroc": roc_auc_score(y[mk], oof[mk])})
    return rows


def run_probe(fold, tr_all, te, R, O, uid_pos, device, inner_only, out):
    base = {"task": fold.task, "compartment": fold.compartment, "fold": fold.fold}
    genes = D.load_genes()
    enc_seeds = CFG["pretrain_run"]["seeds"]
    for v in CFG["pretrain_run"]["variants"]:
        Z = {}
        for s in enc_seeds:
            p = enc_path(fold, v, s)
            if not p.exists():
                raise FileNotFoundError(f"缺少编码器 {p}（先跑 stage 07）")
            model, _ = K.load_encoder(p, CFG, device)
            edges = [e.to(device) for e in D.variant_edges(v, genes, rewired_seed=s % 5, cfg=CFG)]  # 与 07 相同的重连种子
            Z[s] = K.embed(model, R, O, edges, CFG["pretrain_run"]["eval_batch_size"], device, CFG)
        for frac in CFG["eval"]["label_fracs"]:
            seeds = list(range(len(enc_seeds))) if frac >= 1.0 else list(range(CFG["probe"]["seeds"]))
            for seed in seeds:
                Zs = Z[enc_seeds[seed % len(enc_seeds)]]
                tr = D.subsample_labels(tr_all, frac, seed, CFG["eval_ext"]["min_per_class"])
                y, coh = tr.y.to_numpy(), tr.cohort.to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                fit = E.fit_logistic_cv(Zs[uid_pos(tr.sample_uid)], y, coh, CFG["probe"]["C_grid"], "l2",
                                        CFG, CFG["seed"] + seed)
                name = E.model_name(K.VARIANT_MODEL[v], frac)
                out["thr"].append({**base, "model": name, "seed": seed, "C": fit["C"],
                                   "threshold": fit["threshold"], "inner_auc": fit["inner_auc"],
                                   "scheme": fit["scheme"], "n_train": len(tr),
                                   "extra": f"enc_seed={enc_seeds[seed % len(enc_seeds)]}"})
                out["inner"] += _inner_rows(base, name, seed, coh, y, fit["oof"])
                if not inner_only:
                    p = E.predict_logistic(fit, Zs[uid_pos(te.sample_uid)])
                    out["pred"] += [{**base, "model": name, "seed": seed, "sample_uid": u, "y_true": int(t),
                                     "score": float(sc)} for u, t, sc in zip(te.sample_uid, te.y, p)]
        print(f"[08] {fold.key} probe {v} done", flush=True)


def inner_val_split(tr: pd.DataFrame, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    va = []
    for _, g in tr.groupby(["cohort", "y"], sort=True):
        n = int(round(frac * len(g)))
        if len(g) >= 2 and n >= 1:
            va.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    va = np.array(sorted(va), dtype=int)
    return np.setdiff1d(np.arange(len(tr)), va), va


def run_supervised(fold, family, tr_all, te, R, O, uid_pos, device, inner_only, out):
    base = {"task": fold.task, "compartment": fold.compartment, "fold": fold.fold}
    genes = D.load_genes()
    edges = [e.to(device) for e in D.variant_edges("full", genes, cfg=CFG)]
    enc_seeds = CFG["pretrain_run"]["seeds"]
    mname = "B5-scratch" if family == "scratch" else "RRG-full-ft"
    for frac in CFG["eval"]["label_fracs"]:
        for seed in range(CFG["probe"]["seeds"]):
            tr = D.subsample_labels(tr_all, frac, seed, CFG["eval_ext"]["min_per_class"])
            if tr.y.nunique() < 2:
                continue
            tri, vai = inner_val_split(tr, CFG["supervised"]["val_frac"], CFG["seed"] + seed)
            if len(np.unique(tr.y.to_numpy()[vai])) < 2 or len(np.unique(tr.y.to_numpy()[tri])) < 2:
                continue
            set_global_seed(CFG["seed"] + seed)
            if family == "scratch":
                enc = build_encoder(CFG, len(genes), len(edges)).to(device)
            else:
                enc, _ = K.load_encoder(enc_path(fold, "full", enc_seeds[seed % len(enc_seeds)]), CFG, device)
            pos = uid_pos(tr.sample_uid)
            y = tr.y.to_numpy()
            net, info = K.train_supervised(enc, CFG, R[pos], O[pos], y, tri, vai, edges,
                                           finetune=(family == "finetune"), seed=CFG["seed"] + seed,
                                           device=device)
            name = E.model_name(mname, frac)
            thr = E.youden_threshold(y[vai], info["val_prob"])
            out["thr"].append({**base, "model": name, "seed": seed, "C": np.nan, "threshold": thr,
                               "inner_auc": info["val_auc"], "scheme": f"holdout{CFG['supervised']['val_frac']}",
                               "n_train": len(tr), "extra": f"epochs_run={info['epochs_run']}"})
            out["inner"].append({**base, "model": name, "seed": seed, "inner_cohort": "holdout_pooled",
                                 "n": len(vai), "auroc": info["val_auc"]})
            if not inner_only:
                tp = uid_pos(te.sample_uid)
                lg = K.predict_logits(net, R[tp], O[tp], edges, CFG["pretrain_run"]["eval_batch_size"],
                                      device, CFG)
                p = 1 / (1 + np.exp(-lg))
                out["pred"] += [{**base, "model": name, "seed": seed, "sample_uid": u, "y_true": int(t),
                                 "score": float(sc)} for u, t, sc in zip(te.sample_uid, te.y, p)]
        print(f"[08] {fold.key} {family} lf={frac} done", flush=True)


def run_fold(fold: D.Fold, fams: list[str], device, inner_only: bool, stage_dir: Path) -> None:
    set_global_seed(CFG["seed"])
    tr_all, te = D.downstream_split(fold, CFG)
    uids = list(dict.fromkeys(list(tr_all.sample_uid) + list(te.sample_uid)))
    R, O = D.ranks_to_tensors(D.load_ranks(uids))
    pos_of = {u: i for i, u in enumerate(uids)}

    def uid_pos(s):
        return np.array([pos_of[u] for u in s], dtype=int)
    pf = stage_dir / "per_fold"
    pf.mkdir(parents=True, exist_ok=True)
    for fam in fams:
        out = {"pred": [], "thr": [], "inner": []}
        if fam == "probe":
            run_probe(fold, tr_all, te, R, O, uid_pos, device, inner_only, out)
        else:
            run_supervised(fold, fam, tr_all, te, R, O, uid_pos, device, inner_only, out)
        stem = pf / f"{fold.key}__{fam}"
        pd.DataFrame(out["pred"], columns=E.PRED_COLS).to_csv(f"{stem}.pred.tsv", sep="\t", index=False)
        pd.DataFrame(out["thr"], columns=E.THR_COLS).to_csv(f"{stem}.thr.tsv", sep="\t", index=False)
        pd.DataFrame(out["inner"]).to_csv(f"{stem}.inner.tsv", sep="\t", index=False)


def finalize(stage: str, inner_only: bool) -> None:
    sd = ROOT / "results" / stage
    pf = sd / "per_fold"
    stems = [pf / f"{f.key}__{fam}" for f in D.list_folds() if f.task in CFG["tasks"] for fam in families()]
    missing = [s.name for s in stems if not Path(f"{s}.thr.tsv").exists()]
    if missing:
        print(f"[08] 尚缺 {len(missing)} 个 fold×family（不合并、不写 PROVENANCE），例如 {missing[:3]}")
        return
    outs = []
    for kind, name in (("pred", "predictions"), ("thr", "thresholds"), ("inner", "inner_cv")):
        if kind == "pred" and inner_only:
            continue
        parts = [pd.read_csv(f"{s}.{kind}.tsv", sep="\t", dtype={"fold": str}) for s in stems]
        parts = [p for p in parts if len(p)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_csv(sd / f"{name}.tsv", sep="\t", index=False)
        outs.append(sd / f"{name}.tsv")
    proc = ROOT / "data" / "processed"
    enc = sorted({p.resolve() for p in MODELS.glob("*/*/seed*/encoder.pt")})
    write_provenance(stage, [proc / "ranks.parquet", proc / "samples.tsv", proc / "folds.tsv",
                             proc / "genes.txt", *enc], outs, CFG["seed"],
                     {"families": families(), "inner_only": inner_only})
    print(f"[08] 已合并 {len(stems)} 个 fold×family → {sd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--compartment")
    ap.add_argument("--fold")
    ap.add_argument("--families", help="逗号分隔：probe,scratch,finetune")
    ap.add_argument("--inner-only", action="store_true")
    ap.add_argument("--device")
    a = ap.parse_args()
    device = K.device_of(a.device)
    fams = a.families.split(",") if a.families else families()
    stage = "08_models_inner" if a.inner_only else "08_models"
    sd = ROOT / "results" / stage
    for f in D.list_folds(task=a.task, compartment=a.compartment, fold=a.fold):
        if f.task in CFG["tasks"]:
            run_fold(f, fams, device, a.inner_only, sd)
    finalize(stage, a.inner_only)


if __name__ == "__main__":
    main()
