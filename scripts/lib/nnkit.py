"""神经网络训练/推理的共用部件（stage 07 与 08）。"""
from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from lib.model import kidney-injury-repair, SupervisedHead, build_encoder

VARIANT_MODEL = {"full": "RRG-full", "noedge": "B3-noedge", "rewired": "B4-rewired",
                 "ppi_only": "B6-ppi", "coexpr_only": "B6-coexpr"}


def device_of(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(cfg: dict, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["pretrain_run"]["amp_dtype"])
    return torch.autocast("cuda", dtype=dt) if dt else nullcontext()


def batches(n: int, bs: int, rng: np.random.Generator | None, drop_last: bool) -> list[np.ndarray]:
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    out = [idx[i:i + bs] for i in range(0, n, bs)]
    if drop_last and len(out) > 1 and len(out[-1]) < bs:
        out = out[:-1]
    # BatchNorm（VICReg expander）/ VICReg 方差项要求每个 batch ≥ 2
    if out and len(out[-1]) < 2 and len(out) > 1:
        out[-2] = np.concatenate([out[-2], out[-1]])
        out = out[:-1]
    return out


def decay_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    """一维参数（bias、LayerNorm、mask token、逐基因重建偏置）不做 weight decay。"""
    dec = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    nod = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return [{"params": dec, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}]


def cosine_lr(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def embed(model: kidney-injury-repair, R: torch.Tensor, O: torch.Tensor, edges: list[torch.Tensor], bs: int,
          device: torch.device, cfg: dict) -> np.ndarray:
    model.eval()
    zs = []
    for b in batches(len(R), bs, None, False):
        with amp_ctx(cfg, device):
            z = model.embed_samples(R[b].to(device), O[b].to(device), edges)
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs) if zs else np.zeros((0, model.out[0].out_features), np.float32)


def save_encoder(path: Path, model: kidney-injury-repair, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_encoder(path: Path, cfg: dict, device: torch.device) -> tuple[kidney-injury-repair, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_encoder(cfg, meta["n_genes"], meta["n_rel"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device), meta


def _logits(net, R, O, idx, edges, device, cfg):
    with amp_ctx(cfg, device):
        return net(R[idx].to(device), O[idx].to(device), edges).float()


@torch.no_grad()
def predict_logits(net, R, O, edges, bs, device, cfg) -> np.ndarray:
    net.eval()
    out = [_logits(net, R, O, b, edges, device, cfg).cpu().numpy() for b in batches(len(R), bs, None, False)]
    return np.concatenate(out) if out else np.zeros(0)


def train_supervised(encoder: kidney-injury-repair, cfg: dict, R: torch.Tensor, O: torch.Tensor, y: np.ndarray,
                     tr_idx: np.ndarray, va_idx: np.ndarray, edges: list[torch.Tensor],
                     finetune: bool, seed: int, device: torch.device) -> tuple[SupervisedHead, dict]:
    """监督训练（B5：随机初始化；finetune：预训练权重 + 小学习率）。早停看留出集 BCE。

    训练时同样施加平台缺失增强（随机删 drop_frac 的观测基因），与预训练的视图分布一致。
    """
    sc = cfg["supervised"]
    torch.manual_seed(seed)
    net = SupervisedHead(encoder, cfg["model"]["embed_dim"]).to(device)
    if finetune:
        groups = [{"params": net.encoder.parameters(), "lr": sc["lr_finetune_encoder"]},
                  {"params": net.head.parameters(), "lr": sc["lr_finetune_head"]}]
    else:
        groups = [{"params": net.parameters(), "lr": sc["lr_scratch"]}]
    opt = torch.optim.AdamW(groups, weight_decay=sc["weight_decay"])
    base_lr = [g["lr"] for g in opt.param_groups]
    yt = torch.tensor(y, dtype=torch.float32)
    pos = float(y[tr_idx].mean())
    pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    lo, hi = cfg["pretrain"]["drop_frac"]
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(sc["epochs"]):
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * cosine_lr(ep, sc["epochs"], 3)
        net.train()
        tl = []
        for b in batches(len(tr_idx), sc["batch_size"], rng, drop_last=False):
            idx = tr_idx[b]
            Rb, Ob = R[idx].to(device), O[idx].to(device)
            frac = lo + (hi - lo) * torch.rand(len(idx), 1, device=device, generator=gen)
            Ob = Ob & (torch.rand(Ob.shape, device=device, generator=gen) > frac)
            with amp_ctx(cfg, device):
                lg = net(Rb, Ob, edges).float()
            loss = F.binary_cross_entropy_with_logits(lg, yt[idx].to(device), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["pretrain_run"]["grad_clip"])
            opt.step()
            tl.append(float(loss))
        lv = predict_logits(net, R[va_idx], O[va_idx], edges, cfg["pretrain_run"]["eval_batch_size"],
                            device, cfg)
        vl = float(F.binary_cross_entropy_with_logits(torch.tensor(lv), yt[va_idx]))
        hist.append({"epoch": ep, "train_bce": float(np.mean(tl)), "val_bce": vl})
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= sc["patience"]:
                break
    net.load_state_dict(best_state)
    lv = predict_logits(net, R[va_idx], O[va_idx], edges, cfg["pretrain_run"]["eval_batch_size"], device, cfg)
    pv = 1 / (1 + np.exp(-lv))
    yv = y[va_idx]
    auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else float("nan")
    return net, {"val_prob": pv, "val_auc": auc, "best_val_bce": best, "epochs_run": len(hist)}


def timer() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
