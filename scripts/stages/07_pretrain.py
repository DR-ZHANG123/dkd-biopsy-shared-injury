"""Stage 07：每个 (区室, 外部测试队列) 自监督预训练一个编码器。严格不读 diagnosis。

变体：full（PPI + 共表达）/ noedge（B3）/ rewired（B4，保度数重连，seed = 预训练种子 mod 5）/
ppi_only / coexpr_only（B6）。损失 = 遮蔽秩重建 + 平台缺失增强 VICReg（lib/model.py）。
AdamW + warmup-cosine，bf16 AMP，早停看 10% 预训练验证集（按 dup_group 留出，视图随机性固定）。

同一区室、同一测试队列的不同 task（T1/T2）预训练语料相同（按语料哈希判断），只训练一次，
其余 fold 目录以符号链接指向它。

产出：models/<task>_<compartment>_<fold>/<variant>/seed<k>/{encoder.pt, loss.tsv, meta.json}
      results/07_pretrain/runs.tsv、PROVENANCE.json（全部 fold × 变体 × 种子以配置 epoch 完成后才写）
用法：python 07_pretrain.py [--variant full] [--compartment GLOM] [--fold X] [--epochs 3] [--device cuda:0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # 共享服务器：限制 BLAS 线程，避免 joblib/torch 超额订阅

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import nnkit as K  # noqa: E402
from lib.model import build_encoder, gene_mean_rank, gene_mean_recon_baseline, pretrain_loss  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
STAGE = "07_pretrain"
MODELS = ROOT / CFG["paths"]["models"]


def run_dir(fold: D.Fold, variant: str, seed: int) -> Path:
    return MODELS / fold.key / variant / f"seed{seed}"


def _epoch(model, R, O, edges, idx_batches, opt, device, gen, pc) -> dict:
    tot = {"loss": 0.0, "recon": 0.0, "vicreg": 0.0, "n": 0}
    for b in idx_batches:
        Rb, Ob = R[b], O[b]
        with K.amp_ctx(CFG, device):
            loss, parts = pretrain_loss(model, Rb, Ob, edges, CFG["pretrain"]["mask_frac"],
                                        *CFG["pretrain"]["drop_frac"], pc["recon_weight"],
                                        pc["vicreg_weight"], gen=gen)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pc["grad_clip"])
            opt.step()
        n = len(b)
        tot["loss"] += float(loss) * n
        tot["recon"] += float(parts["recon"]) * n
        tot["vicreg"] += float(parts["vicreg"]) * n
        tot["n"] += n
    n = max(tot.pop("n"), 1)
    return {k: v / n for k, v in tot.items()}


def pretrain_one(fold: D.Fold, variant: str, seed: int, epochs: int, device: torch.device,
                 max_batches: int | None) -> dict:
    pc = CFG["pretrain_run"]
    run_seed = CFG["seed"] + 101 * seed
    set_global_seed(run_seed)
    samples = D.load_samples_unlabelled()
    folds = D.load_folds()
    genes = D.load_genes()
    uids = D.pretrain_uids(fold, CFG, samples, folds)
    tr, va = D.split_pretrain_val(uids, samples, CFG["pretrain"]["val_frac"], run_seed)
    Rdf = D.load_ranks(tr + va)
    R, O = D.ranks_to_tensors(Rdf)
    R, O = R.to(device), O.to(device)
    ntr = len(tr)
    tr_idx, va_idx = np.arange(ntr), np.arange(ntr, ntr + len(va))
    edges = [e.to(device) for e in D.variant_edges(variant, genes, rewired_seed=seed % 5, cfg=CFG)]
    gsum = D.graph_summary([e.cpu() for e in edges], len(genes))
    model = build_encoder(CFG, len(genes), len(edges)).to(device)
    model.init_recon_bias(gene_mean_rank(R[tr_idx], O[tr_idx]))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(K.decay_groups(model, CFG["pretrain"]["weight_decay"]), lr=CFG["pretrain"]["lr"])
    base_rec = gene_mean_recon_baseline(R[tr_idx].cpu(), O[tr_idx].cpu(), R[va_idx].cpu(), O[va_idx].cpu())
    rng = np.random.default_rng(run_seed + pc["dataloader_seed_offset"])
    gen = torch.Generator(device=device).manual_seed(run_seed)
    bs = CFG["pretrain"]["batch_size"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        lr = CFG["pretrain"]["lr"] * K.cosine_lr(ep, epochs, pc["warmup_epochs"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = K.timer()
        model.train()
        tb = K.batches(ntr, bs, rng, drop_last=True)[:max_batches]
        trm = _epoch(model, R, O, edges, [tr_idx[b] for b in tb], opt, device, gen, pc)
        model.eval()
        vgen = torch.Generator(device=device).manual_seed(run_seed + 7)   # 固定验证视图
        with torch.no_grad():
            vb = K.batches(len(va_idx), pc["eval_batch_size"], None, drop_last=False)[:max_batches]
            vam = _epoch(model, R, O, edges, [va_idx[b] for b in vb], None, device, vgen, pc)
        hist.append({"epoch": ep, "lr": lr, **{f"train_{k}": v for k, v in trm.items()},
                     **{f"val_{k}": v for k, v in vam.items()}, "sec": K.timer() - t0})
        print(f"[07] {fold.key}/{variant}/s{seed} ep{ep:03d} train={trm['loss']:.4f} val={vam['loss']:.4f} "
              f"rec={vam['recon']:.4f} (gene-mean {base_rec:.4f}) {hist[-1]['sec']:.1f}s", flush=True)
        if vam["loss"] < best - 1e-5:
            best, bad = vam["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= pc["min_epochs"] and bad >= pc["patience"]:
                break
    model.load_state_dict(best_state)
    h = pd.DataFrame(hist)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    meta = {"fold_key": fold.key, "task": fold.task, "compartment": fold.compartment, "fold": fold.fold,
            "test_cohort": fold.test_cohort, "variant": variant, "seed": seed, "run_seed": run_seed,
            "rewired_seed": seed % 5 if variant == "rewired" else None, "n_genes": len(genes),
            "n_rel": len(edges), "graph": gsum, "n_params": n_params, "corpus_hash": D.corpus_hash(uids),
            "n_pretrain": ntr, "n_val": len(va), "epochs_cfg": CFG["pretrain"]["epochs"],
            "epochs_max": epochs, "epochs_run": len(h), "max_batches": max_batches,
            "best_epoch": int(h.val_loss.idxmin()), "best_val_loss": float(best),
            "best_val_recon": float(h.val_recon[h.val_loss.idxmin()]), "gene_mean_recon_mse": base_rec,
            "sec_per_epoch": float(h.sec.mean()), "peak_mem_gib": peak, "device": str(device),
            "val_uids": va}
    d = run_dir(fold, variant, seed)
    d.mkdir(parents=True, exist_ok=True)
    K.save_encoder(d / "encoder.pt", model, {k: meta[k] for k in ("n_genes", "n_rel", "variant", "seed",
                                                                  "corpus_hash", "fold_key")})
    h.to_csv(d / "loss.tsv", sep="\t", index=False)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def link_alias(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            return
    dst.symlink_to(os.path.relpath(src, dst.parent))


def finalize() -> None:
    """全部 fold × 变体 × 种子以配置 epoch 完成时，汇总 runs.tsv 并写 PROVENANCE。"""
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, enc = [], [], []
    for f in D.list_folds():
        if f.task not in CFG["tasks"]:
            continue
        for v in CFG["pretrain_run"]["variants"]:
            for s in CFG["pretrain_run"]["seeds"]:
                mp = run_dir(f, v, s) / "meta.json"
                if not mp.exists():
                    missing.append(f"{f.key}/{v}/s{s}")
                    continue
                m = json.loads(mp.read_text())
                m.pop("val_uids", None)
                m["graph"] = json.dumps(m["graph"])
                m["alias_of"] = "" if m["fold_key"] == f.key else m["fold_key"]
                m["fold_key"] = f.key
                rows.append(m)
                enc.append(run_dir(f, v, s).resolve() / "encoder.pt")
                if m["epochs_max"] != CFG["pretrain"]["epochs"] or m["max_batches"] is not None:
                    missing.append(f"{f.key}/{v}/s{s}(smoke)")
    pd.DataFrame(rows).to_csv(out / "runs.tsv", sep="\t", index=False)
    if missing:
        print(f"[07] 尚未完成 {len(missing)} 个 run（不写 PROVENANCE），例如 {missing[:3]}")
        return
    proc = ROOT / "data" / "processed"
    edge_in = sorted({p for v in CFG["pretrain_run"]["variants"] for s in CFG["pretrain_run"]["seeds"]
                      for p in D.edge_files_for(v, s % 5, CFG)})
    write_provenance(STAGE, [proc / "ranks.parquet", proc / "samples.tsv", proc / "folds.tsv",
                             proc / "genes.txt", *edge_in], [out / "runs.tsv", *sorted(set(enc))],
                     CFG["seed"], {"n_runs": len(rows), "diagnosis_read": False})
    print(f"[07] PROVENANCE 已写：{len(rows)} runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(K.VARIANT_MODEL))
    ap.add_argument("--compartment")
    ap.add_argument("--fold")
    ap.add_argument("--task")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, help="覆盖 config（smoke 用；此类 run 不计入 PROVENANCE）")
    ap.add_argument("--max-batches", type=int, help="每 epoch 最多 batch 数（smoke 用）")
    ap.add_argument("--device")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = K.device_of(a.device)
    variants = [a.variant] if a.variant else CFG["pretrain_run"]["variants"]
    seeds = [a.seed] if a.seed is not None else CFG["pretrain_run"]["seeds"]
    epochs = a.epochs or CFG["pretrain"]["epochs"]
    samples, folds = D.load_samples_unlabelled(), D.load_folds()
    all_folds = [f for f in D.list_folds(folds) if f.task in CFG["tasks"]]
    # 语料哈希 → 规范 fold（按 key 排序的第一个）；跨 task 共享编码器
    canon: dict[tuple, D.Fold] = {}
    for f in sorted(all_folds, key=lambda x: x.key):
        canon.setdefault((f.compartment, D.corpus_hash(D.pretrain_uids(f, CFG, samples, folds))), f)
    target = [f for f in all_folds if (not a.task or f.task == a.task)
              and (not a.compartment or f.compartment == a.compartment) and (not a.fold or f.fold == a.fold)]
    for f in sorted(target, key=lambda x: x.key):
        c = canon[(f.compartment, D.corpus_hash(D.pretrain_uids(f, CFG, samples, folds)))]
        for v in variants:
            for s in seeds:
                src = run_dir(c, v, s)
                if not (src / "encoder.pt").exists() or (a.force and c == f):
                    if c != f and not (src / "encoder.pt").exists():
                        print(f"[07] {f.key} 与 {c.key} 语料相同 → 训练于 {c.key}")
                    pretrain_one(c, v, s, epochs, device, a.max_batches)
                if c != f:
                    link_alias(src, run_dir(f, v, s))
    finalize()


if __name__ == "__main__":
    main()
