"""
mae.py — Group E：MAE 预训练 Transformer 编码器（M14）

基于 Masked Autoencoder（He et al., CVPR 2022）应用于 DNA 序列：
  预训练阶段：
    随机遮蔽 75% 的 token，编码器（M6 架构）处理未遮蔽部分，
    轻量解码器（2 层 Transformer）重建被遮蔽 token 的原始独热向量（MSE loss）。

  微调阶段：
    丢弃解码器，编码器权重用预训练结果初始化，
    接融合模块和分类头，进行有监督 EPI 分类。

与 LLM 编码器（M13）的对比：
  M13 — 使用外部大规模预训练权重（无需本地数据）
  M14 — 在目标细胞系数据上自监督预训练，分布更匹配，内存需求更低
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


# ── 正弦位置编码（复用，避免循环导入）────────────────────────────────────────

class _SinPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


# ── MAE 编码器（与 M6 共享相同架构）─────────────────────────────────────────

class MAEEncoder(nn.Module):
    """
    MAE 编码器，与 M6_Transformer 结构完全相同。
    CNN 前端降采样 + 4 层 TransformerEncoderLayer + CLS token。
    预训练完成后，编码器权重可直接用于 M14_MAE_Transformer。
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # CNN 前端：5 → d_model，stride=16
        self.cnn_front = nn.Sequential(
            nn.Conv1d(5, d_model, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        # 可学习 mask token（遮蔽位置用这个向量填充）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pe = _SinPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, mask_ratio: float = 0.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, 5, L) → (h, mask, ids_restore)
        预训练时 mask_ratio=0.75；微调时 mask_ratio=0.0（全部可见）。

        返回：
          h          — 编码后的可见 token（含 CLS），(B, n_visible+1, d_model)
          mask       — 布尔 mask 矩阵，True=被遮蔽，(B, L')
          ids_restore — 用于解码器重建位置顺序的索引，(B, L')
        """
        # CNN 降采样：(B, 5, L) → (B, L', d_model)
        h = self.cnn_front(x).permute(0, 2, 1)     # (B, L', d_model)
        B, Lp, d = h.shape

        if mask_ratio > 0.0:
            # 随机遮蔽
            n_mask = int(Lp * mask_ratio)
            noise = torch.rand(B, Lp, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, n_mask:]       # 可见 token 索引
            h_visible = torch.gather(
                h, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
            )                                        # (B, Lp-n_mask, d)
            # 生成 mask：1=被遮蔽，0=可见
            mask = torch.ones(B, Lp, device=x.device)
            mask.scatter_(1, ids_keep, 0.0)
        else:
            h_visible = h
            ids_restore = torch.arange(Lp, device=x.device).unsqueeze(0).expand(B, -1)
            mask = torch.zeros(B, Lp, device=x.device)

        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h_visible = torch.cat([cls, h_visible], dim=1)
        h_visible = self.pe(h_visible)
        h_visible = self.transformer(h_visible)
        h_visible = self.norm(h_visible)

        return h_visible, mask, ids_restore


# ── MAE 解码器（预训练专用，微调时丢弃）──────────────────────────────────────

class MAEDecoder(nn.Module):
    """
    轻量 MAE 解码器（2 层 Transformer）。
    重建目标：被遮蔽 token 对应位置的原始独热向量（MSE loss）。
    """

    def __init__(self, encoder_dim: int = 256, decoder_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 512, dropout: float = 0.1,
                 cnn_stride: int = 16):
        super().__init__()
        # 编码器维度到解码器维度的投影
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.pe = _SinPE(decoder_dim)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(decoder_dim)
        # 输出每个下采样位置对应的 cnn_stride × 5 个独热值（MSE 重建）
        self.pred_head = nn.Linear(decoder_dim, cnn_stride * 5)

    def forward(
        self, h: Tensor, ids_restore: Tensor, mask_token: nn.Parameter
    ) -> Tensor:
        """
        h: (B, n_visible+1, encoder_dim)
        ids_restore: (B, Lp)
        返回：predicted (B, Lp, cnn_stride*5)
        """
        B = h.size(0)
        h = self.proj(h)                            # (B, n_vis+1, decoder_dim)

        # 去掉 CLS token
        h_tokens = h[:, 1:]                         # (B, n_visible, decoder_dim)
        n_vis = h_tokens.size(1)
        Lp = ids_restore.size(1)

        # 用 mask_token 填充被遮蔽位置
        mask_tokens = self.proj(mask_token).expand(B, Lp - n_vis, -1)
        h_full = torch.cat([h_tokens, mask_tokens], dim=1)  # (B, Lp, decoder_dim)

        # 按 ids_restore 还原顺序
        h_full = torch.gather(
            h_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, h_full.size(-1))
        )

        h_full = self.pe(h_full)
        h_full = self.transformer(h_full)
        return self.pred_head(self.norm(h_full))    # (B, Lp, cnn_stride*5)


# ── M14：MAE 预训练 Transformer 编码器 ────────────────────────────────────────

class M14_MAE_Transformer(BaseEPIModel):
    """
    MAE 预训练 Transformer 编码器（M14）。

    两阶段训练流程：
      1. pretrain_step(x_unlabeled)：遮蔽 75% token，MSE 重建，更新 encoder + decoder
      2. forward(x_e, x_p)：标准 EPI 分类前向（解码器参数不参与）

    预训练完成后可通过 save_encoder() / load_pretrained() 保存/加载编码器权重。
    """

    D_ENC = 256
    MASK_RATIO = 0.75

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        self.encoder = MAEEncoder(
            d_model=self.D_ENC,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        # 解码器仅在预训练阶段使用
        self.decoder = MAEDecoder(
            encoder_dim=self.D_ENC,
            decoder_dim=128,
            cnn_stride=16,
        )

    def encode(self, x: Tensor) -> Tensor:
        """
        微调阶段：mask_ratio=0.0，全部 token 可见，取 CLS 表示。
        x: (B, 5, L) → (B, D_ENC)
        """
        h, _, _ = self.encoder(x, mask_ratio=0.0)
        return h[:, 0]                              # CLS → (B, D_ENC)

    def pretrain_step(self, x: Tensor) -> Tensor:
        """
        预训练前向传播（train.py 中 --pretrain 模式调用）。
        x: (B, 5, L)
        返回重建 MSE loss（标量）。
        """
        B, _, L = x.shape

        h, mask, ids_restore = self.encoder(x, mask_ratio=self.MASK_RATIO)

        # 解码器预测
        pred = self.decoder(h, ids_restore, self.encoder.mask_token)
        # (B, Lp, 16*5) 中仅对 mask=1 的位置计算 loss

        # 构建 ground truth：CNN 前端对应的 patch（stride=16 个碱基对的独热向量）
        # 下采样后每个 patch = 16 × 5 个 one-hot 值（拉平）
        x_padded = x[:, :, : (L // 16) * 16]        # 对齐到 stride 倍数
        target = x_padded.unfold(2, 16, 16)          # (B, 5, Lp, 16)
        target = target.permute(0, 2, 1, 3).reshape(B, -1, 80)  # (B, Lp, 80)

        Lp = target.size(1)
        pred = pred[:, :Lp]

        # 仅对被遮蔽的 patch 计算 MSE
        loss = ((pred - target) ** 2).mean(dim=-1)   # (B, Lp)
        mask_used = mask[:, :Lp]
        loss = (loss * mask_used).sum() / (mask_used.sum() + 1e-8)
        return loss

    def save_encoder(self, path: str) -> None:
        """保存编码器权重（微调时加载）。"""
        torch.save(self.encoder.state_dict(), path)

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """从预训练 checkpoint 加载编码器权重。"""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=strict)
        print(f"[M14] 编码器预训练权重已加载：{path}")
