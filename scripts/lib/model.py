"""dkd-biopsy-shared-injury 编码器：样本内秩 → 基因图（多关系 GATv2）→ attention pooling → 样本表示。

同一张基因图被一个 batch 内的所有样本共享；用 PyG 的 batch 偏移方式堆叠（按 batch 大小缓存）。
缺失基因（平台测不到或被遮蔽）以可学习的 mask token 表示，不以 0 表示。
edges 为空列表时即对照 B3（无边，等价于 DeepSets）。

相对初稿的修改（理由见 notes/01_execution_log.md）：
1. 未观测位置的秩先置 0 再进 rank_mlp —— 初稿把 NaN 送进 MLP，torch.where 虽在前向屏蔽，
   反向时 0×NaN 会让 rank_mlp 的权重梯度变成 NaN；
2. VICReg 作用在独立的 expander（proj_dim）上，下游 embedding 仍是 64 维 z（VICReg 原文做法，
   避免方差/协方差项直接把 64 维表示压成白化坐标）；
3. batch 边索引按 (B, 设备) 缓存，避免每步重建 B×E 的索引；
4. pretrain_loss 支持外部 torch.Generator（验证集用固定视图，使早停所看的验证损失可复现）；
5. 重建头加逐基因偏置并初始化为预训练语料的逐基因平均秩（recon 权重初始化为 0）：
   秩的方差大部分是「基因身份」带来的（gene-mean 基线 MSE 远低于常数预测），不加偏置时
   前若干 epoch 的重建梯度都花在记忆基因均值上；加偏置后重建损失只衡量样本特异的偏离；
6. readout 可选 attention（方案原设，默认）/ genewise / hybrid（config model_ext.readout），见类内注释；
7. 可选全局上下文消息（config model_ext.global_context）：无边时被遮蔽节点没有任何信息来源，
   B3 的重建目标退化为常数（实测真实数据上 B3 重建 MSE 始终等于 gene-mean 基线），
   加入后 B3 = DeepSets + set context，与 full 只差图边；
8. 增加 SupervisedHead，供 B5（不预训练）与全参微调复用同一编码器。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class RankEmbed(nn.Module):
    """秩标量 → d 维；与基因身份 embedding 相加。"""

    def __init__(self, n_genes: int, d: int):
        super().__init__()
        self.gene = nn.Embedding(n_genes, d)
        nn.init.normal_(self.gene.weight, std=0.1)
        self.rank_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.mask_token = nn.Parameter(torch.zeros(d))

    def forward(self, ranks: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        # ranks, observed: (B, N)；未观测位置置 0 —— 防止 NaN 通过反向传播污染 rank_mlp 梯度
        r = torch.where(observed, ranks, torch.zeros_like(ranks))
        h = self.rank_mlp(r.unsqueeze(-1))
        h = torch.where(observed.unsqueeze(-1), h, self.mask_token.to(h.dtype).expand_as(h))
        return h + self.gene.weight.unsqueeze(0).to(h.dtype)


class RelGATLayer(nn.Module):
    """每种边类型一个 GATv2Conv，输出相加，残差 + LayerNorm。"""

    def __init__(self, d: int, heads: int, n_rel: int, dropout: float, global_context: bool = False):
        super().__init__()
        self.convs = nn.ModuleList(
            GATv2Conv(d, d // heads, heads=heads, dropout=dropout, add_self_loops=False)
            for _ in range(n_rel))
        self.self_lin = nn.Linear(d, d)
        # 全局上下文：样本内全部节点的均值广播回每个节点（DeepSets 式 set context）。
        # 所有变体一致使用，B3（无边）因此仍能从样本整体状态重建被遮蔽基因，B3 与 full 只差图边。
        self.ctx_lin = nn.Linear(d, d) if global_context else None
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edges: list[torch.Tensor], B: int) -> torch.Tensor:
        out = self.self_lin(x)
        if self.ctx_lin is not None:
            xb = x.reshape(B, -1, x.shape[-1])
            out = out + self.ctx_lin(xb.mean(1, keepdim=True)).expand_as(xb).reshape(x.shape)
        for conv, ei in zip(self.convs, edges):
            if ei.numel():
                out = out + conv(x, ei)
        return self.norm(x + self.drop(F.gelu(out)))


class dkd-biopsy-shared-injury(nn.Module):
    def __init__(self, n_genes: int, n_rel: int, d: int = 64, layers: int = 3, heads: int = 4,
                 dropout: float = 0.1, embed_dim: int = 64, proj_dim: int = 256,
                 readout: str = "attention", readout_channels: int = 2, global_context: bool = False):
        super().__init__()
        self.n_genes = n_genes
        self.n_rel = n_rel
        self.readout = readout
        self.embed = RankEmbed(n_genes, d)
        # n_rel = 0（B3）时不建 conv，参数量与信息流都等价于逐基因 MLP + attention pooling
        self.layers = nn.ModuleList(RelGATLayer(d, heads, n_rel, dropout, global_context)
                                    for _ in range(layers))
        self.pool_gate = nn.Linear(d, 1)
        # readout：attention = 方案原设（共享门控的加权平均）；genewise = 每个节点压成 k 个通道后
        # 用基因特异的线性映射汇总（可精确表达「平滑后秩的线性组合」，因此表达能力包含线性基线）；
        # hybrid = 二者拼接。
        n_in = {"attention": d, "genewise": embed_dim, "hybrid": d + embed_dim}[readout]
        if readout in ("genewise", "hybrid"):
            self.node_score = nn.Linear(d, readout_channels)
            self.gene_read = nn.Linear(n_genes * readout_channels, embed_dim)
        self.out = nn.Sequential(nn.Linear(n_in, embed_dim), nn.LayerNorm(embed_dim))
        self.recon = nn.Linear(d, 1)          # 遮蔽秩重建头
        self.recon_bias = nn.Parameter(torch.zeros(n_genes))   # 逐基因偏置，初始化为语料平均秩
        self.proj = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
                                  nn.Linear(proj_dim, proj_dim))  # VICReg expander（仅预训练）
        self._edge_cache: dict = {}

    def batch_edges(self, edges: list[torch.Tensor], B: int, N: int) -> list[torch.Tensor]:
        if not edges:
            return []
        dev = edges[0].device
        key = (B, N, str(dev), tuple(int(e.data_ptr()) for e in edges))
        if key not in self._edge_cache:
            off = torch.arange(B, device=dev) * N
            self._edge_cache = {key: [(ei.unsqueeze(0) + off.view(B, 1, 1)).permute(1, 0, 2).reshape(2, -1)
                                      for ei in edges]}
        return self._edge_cache[key]

    def node_states(self, ranks, observed, edges):
        B, N = ranks.shape
        h = self.embed(ranks, observed).reshape(B * N, -1)
        be = self.batch_edges(edges, B, N)
        for layer in self.layers:
            h = layer(h, be, B)
        return h.reshape(B, N, -1)

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.readout in ("attention", "hybrid"):
            a = self.pool_gate(h).squeeze(-1).float()                  # (B, N)
            w = torch.softmax(a, dim=1).to(h.dtype)
            parts.append((w.unsqueeze(-1) * h).sum(1))
        if self.readout in ("genewise", "hybrid"):
            s = self.node_score(h).flatten(1)                          # (B, N·k)
            parts.append(self.gene_read(s))
        return self.out(torch.cat(parts, -1) if len(parts) > 1 else parts[0])

    def forward(self, ranks, observed, edges):
        h = self.node_states(ranks, observed, edges)
        return self.pool(h), h

    @torch.no_grad()
    def init_recon_bias(self, gene_mean: torch.Tensor) -> None:
        self.recon_bias.copy_(gene_mean.to(self.recon_bias))
        nn.init.zeros_(self.recon.weight)
        nn.init.zeros_(self.recon.bias)

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        return self.recon(h).squeeze(-1).float() + self.recon_bias

    @torch.no_grad()
    def embed_samples(self, ranks, observed, edges) -> torch.Tensor:
        return self.forward(ranks, observed, edges)[0]


class SupervisedHead(nn.Module):
    """编码器 + 线性 logit 头（B5 从随机初始化训练；RRG-full-ft 从预训练权重小学习率微调）。"""

    def __init__(self, encoder: dkd-biopsy-shared-injury, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, ranks, observed, edges):
        z, _ = self.encoder(ranks, observed, edges)
        return self.head(z).squeeze(-1)


def vicreg(z1: torch.Tensor, z2: torch.Tensor, sim_w=25.0, var_w=25.0, cov_w=1.0
           ) -> tuple[torch.Tensor, dict]:
    z1, z2 = z1.float(), z2.float()
    inv = F.mse_loss(z1, z2)

    def var_cov(z):
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        v = F.relu(1 - std).mean()
        c = (z.T @ z) / (len(z) - 1)
        off = c - torch.diag(torch.diag(c))
        return v, (off ** 2).sum() / z.shape[1]
    v1, c1 = var_cov(z1)
    v2, c2 = var_cov(z2)
    loss = sim_w * inv + var_w * (v1 + v2) + cov_w * (c1 + c2)
    return loss, {"inv": inv.detach(), "var": (v1 + v2).detach(), "cov": (c1 + c2).detach()}


def pretrain_loss(model: dkd-biopsy-shared-injury, ranks, observed, edges, mask_frac: float,
                  drop_lo: float, drop_hi: float, recon_w: float = 10.0, vic_w: float = 0.04,
                  gen: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """(a) 遮蔽秩重建 + (b) 平台缺失增强下的 VICReg。返回的 dict 值为 0 维张量（调用方再同步）。"""
    B, N = ranks.shape
    dev = ranks.device

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=gen)

    def view():
        frac = drop_lo + (drop_hi - drop_lo) * rand(B, 1)
        return observed & (rand(B, N) > frac)

    o1, o2 = view(), view()
    m = (rand(B, N) < mask_frac) & o1
    z1, h1 = model(ranks, o1 & ~m, edges)
    z2, _ = model(ranks, o2, edges)
    pred = model.reconstruct(h1)
    l_rec = F.mse_loss(pred[m], ranks[m].float()) if bool(m.any()) else pred.sum() * 0
    l_vic, parts = vicreg(model.proj(z1.float()), model.proj(z2.float()))
    loss = recon_w * l_rec + vic_w * l_vic
    return loss, {"recon": l_rec.detach(), "vicreg": l_vic.detach(), **parts}


def gene_mean_rank(ranks: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    s = (ranks * obs).sum(0)
    c = obs.sum(0).clamp(min=1)
    return torch.where(obs.sum(0) > 0, s / c, torch.full_like(s, 0.5))


def gene_mean_recon_baseline(ranks_tr: torch.Tensor, obs_tr: torch.Tensor,
                             ranks_va: torch.Tensor, obs_va: torch.Tensor) -> float:
    """诊断用：用训练集逐基因平均秩预测验证集秩的 MSE —— 重建损失必须低于它才说明学到了样本特异信息。"""
    mu = gene_mean_rank(ranks_tr, obs_tr)
    err = ((ranks_va - mu.unsqueeze(0)) ** 2)[obs_va]
    return float(err.mean()) if err.numel() else float("nan")


def build_encoder(cfg: dict, n_genes: int, n_rel: int) -> dkd-biopsy-shared-injury:
    m = cfg["model"]
    return dkd-biopsy-shared-injury(n_genes, n_rel, d=m["hidden"], layers=m["layers"], heads=m["heads"],
                        dropout=m["dropout"], embed_dim=m["embed_dim"],
                        proj_dim=cfg["pretrain_run"]["proj_dim"],
                        readout=cfg.get("model_ext", {}).get("readout", "attention"),
                        readout_channels=cfg.get("model_ext", {}).get("readout_channels", 2),
                        global_context=cfg.get("model_ext", {}).get("global_context", False))
