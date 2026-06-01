"""
hybrid.py — Group E：混合编码器（M11 / M12）

M11 — CNN + BiLSTM
  M1 的 CNN 前端（降采样至 ~500 token）作为局部 motif 检测器，
  再接 M4 的双向 LSTM 捕获长程依赖。

M12 — CNN + Transformer
  同 M11 但将 BiLSTM 替换为标准 Transformer Encoder（M6）。
  CNN 降采样解决 Transformer 的 O(L²) 问题。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


# ── 共用 CNN 前端（截断版，输出序列而非全局向量）────────────────────────────

class _CNNFrontend(nn.Module):
    """
    CNN 局部特征提取器，输出序列形式（保留时间维度）供后续 RNN/Transformer 使用。
    两阶段卷积 + 池化后，序列长度约为 L // (4*4*2*2) = L // 64 ≈ 156（L=10000 时）。
    """

    def __init__(self, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 阶段 1：5 → 64，stride=4，MaxPool(2)
            nn.Conv1d(5, 64, kernel_size=24, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # 阶段 2：64 → 128，stride=4，MaxPool(2)
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # 阶段 3：128 → d_out，不降采样
            nn.Conv1d(128, d_out, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, L'', d_out)，L'' ≈ L // 64"""
        return self.net(x).permute(0, 2, 1)   # (B, L'', d_out)


# ── M11：CNN + BiLSTM ─────────────────────────────────────────────────────────

class M11_CNN_BiLSTM(BaseEPIModel):
    """
    CNN + BiLSTM 编码器（M11）。
    CNN 前端提取局部 motif 特征序列，BiLSTM 在压缩后的序列上捕获全局依赖。
    """

    D_ENC = 512

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        D_CNN = 256
        self.cnn = _CNNFrontend(d_out=D_CNN)
        self.lstm = nn.LSTM(
            input_size=D_CNN,
            hidden_size=256,           # 双向后 = 512
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, 512)"""
        h = self.cnn(x)                            # (B, L'', D_CNN)
        _, (h_n, _) = self.lstm(h)
        # 最后一层前向/后向拼接
        h_fwd = h_n[-2]                            # (B, 256)
        h_bwd = h_n[-1]                            # (B, 256)
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 512)


# ── M12：CNN + Transformer ────────────────────────────────────────────────────

class M12_CNN_Transformer(BaseEPIModel):
    """
    CNN + Transformer 编码器（M12）。
    CNN 前端压缩序列，Transformer 在 ~156 个 token 上做全局 attention（O(156²)≈24k 步）。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        self.cnn = _CNNFrontend(d_out=self.D_ENC)

        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.D_ENC))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Embedding(2048, self.D_ENC)  # 覆盖最大 token 数

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_ENC,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        h = self.cnn(x)                            # (B, L'', D_ENC)
        B, Lp, _ = h.shape

        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)             # (B, L''+1, D_ENC)

        # 位置编码（可学习）
        pos = torch.arange(h.size(1), device=x.device)
        h = h + self.pos_embed(pos).unsqueeze(0)

        h = self.transformer(h)
        h = self.norm(h)
        return h[:, 0]                             # CLS 位置 → (B, D_ENC)
