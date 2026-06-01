"""
transformer.py — Group C：Transformer 族编码器（M6 / M7 / M8）

M6 — 标准 Transformer Encoder
  CNN 前端（stride=16）将序列降采样至 ~625 token，
  再接 4 层 TransformerEncoderLayer（d=256, head=8）+ CLS token。

M7 — Linear Transformer（Katharopoulos et al., ICML 2020）
  ELU+1 核函数近似 softmax attention，复杂度 O(L d²)，
  无需 CNN 降采样，直接处理 10 kbp 单碱基分辨率序列。

M8 — iTransformer（Liu et al., ICLR 2024）
  在 C=5 通道维（ACGTN）而非位置维做 attention，
  复杂度 O(C² L) = O(25L)，天然适合 one-hot 输入格式。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


# ── 正弦位置编码 ──────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """标准正弦/余弦位置编码（Vaswani et al., 2017），支持最大 50k 个 token。"""

    def __init__(self, d_model: int, max_len: int = 50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # 注册为 buffer（不参与梯度更新，但随模型保存/迁移设备）
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model)"""
        return x + self.pe[:, : x.size(1)]


# ── M6：标准 Transformer Encoder ─────────────────────────────────────────────

class M6_Transformer(BaseEPIModel):
    """
    标准 Transformer 编码器（M6）。
    CNN 前端（Conv1d stride=16）将 L=10000 降至 ~625 token，
    再接 4 层 TransformerEncoderLayer，CLS token 作为序列表示。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        # CNN 前端：5 → d_model，stride=16 降采样
        self.cnn_front = nn.Sequential(
            nn.Conv1d(5, self.D_ENC, kernel_size=16, stride=16, padding=0),
            nn.BatchNorm1d(self.D_ENC),
            nn.ReLU(),
        )
        # 可学习 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.D_ENC))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pe = SinusoidalPE(self.D_ENC)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_ENC,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        # CNN 降采样：(B, 5, L) → (B, D_ENC, L//16)
        h = self.cnn_front(x).permute(0, 2, 1)     # (B, L', D_ENC)

        # 拼接 CLS token
        B = h.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)              # (B, L'+1, D_ENC)

        h = self.pe(h)
        h = self.transformer(h)
        h = self.norm(h)
        return h[:, 0]                              # CLS 位置 → (B, D_ENC)


# ── M7：Linear Transformer ────────────────────────────────────────────────────

class _LinearAttention(nn.Module):
    """
    ELU+1 核函数线性 attention（Katharopoulos et al., ICML 2020）：
      ϕ(x) = ELU(x) + 1
      Attn_i = ϕ(q_i)^T (Σ_j ϕ(k_j) v_j^T) / (ϕ(q_i)^T Σ_j ϕ(k_j))
    复杂度 O(L d²)，消除二次项。
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _kernel(x: Tensor) -> Tensor:
        """ϕ(x) = ELU(x) + 1，保证非负性。"""
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, d = x.shape
        H, Dh = self.n_heads, self.d_head

        def reshape(t: Tensor) -> Tensor:
            # (B, L, d) → (B, H, L, Dh)
            return t.view(B, L, H, Dh).transpose(1, 2)

        Q = self._kernel(reshape(self.q_proj(x)))  # (B, H, L, Dh)
        K = self._kernel(reshape(self.k_proj(x)))  # (B, H, L, Dh)
        V = reshape(self.v_proj(x))                # (B, H, L, Dh)

        # 核心：KV 先聚合（O(L Dh²)）再与 Q 相乘（O(L Dh²)）
        KV = torch.einsum("bhld,bhlv->bhdv", K, V)  # (B, H, Dh, Dh)
        QKV = torch.einsum("bhld,bhdv->bhlv", Q, KV)  # (B, H, L, Dh)

        # 归一化分母
        K_sum = K.sum(dim=2)                       # (B, H, Dh)
        denom = torch.einsum("bhld,bhd->bhl", Q, K_sum).unsqueeze(-1) + 1e-6

        out = (QKV / denom).transpose(1, 2).contiguous().view(B, L, d)
        return self.out_proj(out)


class M7_LinearTransformer(BaseEPIModel):
    """
    Linear Transformer 编码器（M7）。
    无需 CNN 前端，直接在单碱基分辨率处理序列。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        # 输入投影：5 → D_ENC（逐位置线性，等效于 1×1 卷积）
        self.input_proj = nn.Linear(5, self.D_ENC)
        self.pe = SinusoidalPE(self.D_ENC)
        # 叠加 n_layers 层 Linear Attention + FFN
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": _LinearAttention(self.D_ENC, config.n_heads),
                "ffn":  nn.Sequential(
                    nn.LayerNorm(self.D_ENC),
                    nn.Linear(self.D_ENC, config.d_ff),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_ff, self.D_ENC),
                ),
                "norm": nn.LayerNorm(self.D_ENC),
            })
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, D_ENC)"""
        # (B, 5, L) → (B, L, 5) → (B, L, D_ENC)
        h = self.input_proj(x.permute(0, 2, 1))
        h = self.pe(h)
        for layer in self.layers:
            h = h + layer["attn"](layer["norm"](h))   # Pre-LN + 残差
            h = h + layer["ffn"](h)
        return self.final_norm(h).mean(dim=1)          # 均值池化 → (B, D_ENC)


# ── M8：iTransformer ──────────────────────────────────────────────────────────

class M8_iTransformer(BaseEPIModel):
    """
    iTransformer（M8，Liu et al., ICLR 2024）。
    原始 iTransformer 用于时间序列（在 variate 维做 attention）。
    此处将 C=5 核苷酸通道视为 "variate"，在通道维做 attention，
    FFN 在位置维捕获序列内非线性模式。
    复杂度 O(C² L) = O(25L)，远低于标准 O(L²)。
    """

    D_ENC = 256

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        # 在 C=5 通道维做多头 attention（每个 head 对应不同核苷酸组合关系）
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=5,          # Q/K/V 都是 5 维（通道数）
            num_heads=1,          # 通道数小，单头足够
            dropout=config.dropout,
            batch_first=True,
        )
        # FFN 在位置维处理每个通道的非线性特征（等效于 1D 逐位置变换）
        self.ffn = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 5),
        )
        self.norm = nn.LayerNorm(5)
        # 将 (5, L) 展平后投影到 D_ENC
        self.output_proj = nn.Linear(5, self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """
        x: (B, 5, L) → (B, D_ENC)
        在通道维（5 个核苷酸类型）做 self-attention，捕获共现统计；
        在位置维做 FFN，捕获逐位置非线性。
        """
        B, C, L = x.shape

        # 通道 attention：将 (B, C, L) 视为序列长度=L，特征维=C
        # 转置为 (B, L, C)，在 C 维做 attention（等效于逐位置对 C 通道 attend）
        h = x.permute(0, 2, 1)                         # (B, L, C=5)
        h_attn, _ = self.channel_attn(h, h, h)         # (B, L, 5)
        h = self.norm(h + h_attn)                       # 残差 + LayerNorm

        # FFN
        h = h + self.ffn(h)                             # (B, L, 5)

        # 全局均值池化 + 投影
        h = h.mean(dim=1)                               # (B, 5)
        return self.output_proj(h)                      # (B, D_ENC)
