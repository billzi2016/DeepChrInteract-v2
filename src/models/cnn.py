"""
cnn.py — Group A：经典 CNN 编码器（M1 / M2 / M3）

复现原始 DeepChrInteract (Keras) 中的 CNN 架构，用 PyTorch 重写：
  M1 — CNN 单路（复现 model_onehot_cnn_one_branch）
  M2 — CNN 双路（复现 model_onehot_cnn_two_branch）
  M3 — k-mer Embedding + CNN（复现 model_embedding_cnn_two_branch）

输入：
  M1/M2 — (B, 5, L) one-hot 独热张量
  M3    — (B, L')  k-mer token 序列，L' = L - k + 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel


# ── 共用 CNN 骨干 ─────────────────────────────────────────────────────────────

class _CNNBackbone(nn.Module):
    """
    两阶段 Conv1d 骨干：
      阶段 1：Conv1d(in_ch→64, k=24, s=4) + BN + ReLU + MaxPool(2)
      阶段 2：Conv1d(64→128, k=24, s=4) + BN + ReLU + MaxPool(2)
    全局平均池化 → Linear(128→512) → Dropout(0.5) → Linear(512→d_out)
    """

    def __init__(self, in_channels: int = 5, d_out: int = 512):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=24, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=24, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # 全局平均池化 + 两层 FC（等效于原始 Keras Flatten + Dense）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # → (B, 128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_channels, L) → (B, d_out)"""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x).squeeze(-1)   # (B, 128)
        return self.fc(x)              # (B, d_out)


# ── M1：CNN 单路 ──────────────────────────────────────────────────────────────

class M1_CNN_SingleBranch(BaseEPIModel):
    """
    单路 CNN（M1）：仅编码增强子序列，不使用融合模块。
    复现 model_onehot_cnn_one_branch，验证最基础的 baseline。
    """

    D_ENC = 512

    def __init__(self, config: Config):
        # 单分支：fusion 使用 concat（d_fused = 2*512 = 1024），但实际跳过融合
        # 为保持接口一致，dual_branch=False 时基类直接将 h_e 送入分类头
        super().__init__(config, self.D_ENC, dual_branch=False)
        self.backbone = _CNNBackbone(in_channels=5, d_out=self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, 512)"""
        return self.backbone(x)


# ── M2：CNN 双路 ──────────────────────────────────────────────────────────────

class M2_CNN_DualBranch(BaseEPIModel):
    """
    双路 CNN（M2）：增强子/启动子各用一个独立 CNN 编码器，再融合。
    复现 model_onehot_cnn_two_branch。
    参数量约为 M1 的 2 倍。
    """

    D_ENC = 512

    def __init__(self, config: Config):
        super().__init__(config, self.D_ENC, dual_branch=True)
        self.backbone = _CNNBackbone(in_channels=5, d_out=self.D_ENC)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 5, L) → (B, 512)"""
        return self.backbone(x)


# ── M3：k-mer Embedding + CNN ─────────────────────────────────────────────────

class M3_Kmer_CNN(BaseEPIModel):
    """
    k-mer 嵌入 + CNN（M3）。
    输入为 k-mer token 序列 (B, L')，先通过可学习 Embedding(4097, 128)，
    再经过 4 个 Conv1d(k=32, s=8) + ReLU + MaxPool 块，最后全局平均池化。
    复现 model_embedding_cnn_two_branch。
    """

    VOCAB_SIZE = 4097   # 4^6 + padding(0) + unk
    EMBED_DIM  = 128

    def __init__(self, config: Config):
        D_ENC = 256
        super().__init__(config, D_ENC, dual_branch=True)
        # 可学习词嵌入；padding_idx=0 对应 pad token
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.EMBED_DIM, padding_idx=0)
        # 4 个 CNN 块（通道数从 EMBED_DIM 开始逐步扩展）
        self.conv_blocks = nn.Sequential(
            self._conv_block(self.EMBED_DIM, 64,  kernel_size=32, stride=8),
            self._conv_block(64,            64,  kernel_size=16, stride=4),
            self._conv_block(64,            128, kernel_size=8,  stride=2),
            self._conv_block(128,           D_ENC, kernel_size=4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel_size: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, L') LongTensor → (B, D_ENC)"""
        # 嵌入：(B, L') → (B, L', EMBED_DIM) → (B, EMBED_DIM, L') 供 Conv1d
        h = self.embedding(x).transpose(1, 2)   # (B, 128, L')
        h = self.conv_blocks(h)                  # (B, D_ENC, L'')
        return self.pool(h).squeeze(-1)          # (B, D_ENC)
