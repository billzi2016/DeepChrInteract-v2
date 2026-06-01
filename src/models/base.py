"""
base.py — BaseEPIModel 抽象基类 + 共享分类头

所有 14 个编码器模型均继承此类，并实现 encode() 方法。
统一接口：forward(x_e, x_p) → logit (未经 sigmoid)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from .fusion import FusionModule


class ClassificationHead(nn.Module):
    """
    两层 MLP 分类头：
      Linear(d_in, 256) → ReLU → Dropout → Linear(256, 1)
    输出未经 sigmoid 的 logit（与 BCEWithLogitsLoss 配合使用）。
    """

    def __init__(self, d_in: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class BaseEPIModel(nn.Module, ABC):
    """
    EPI 预测模型基类。

    子类只需实现 encode(x) → Tensor(B, d_enc)，
    融合策略和分类头由基类统一处理。

    单分支模式（M1）：encode(x_e) 直接接分类头，忽略 x_p。
    双分支模式（M2–M14）：encode(x_e) 和 encode(x_p) 分别编码后融合。
    """

    def __init__(self, config: Config, d_enc: int, dual_branch: bool = True):
        super().__init__()
        self.dual_branch = dual_branch
        if dual_branch:
            # 双路模式：h_e 和 h_p 经融合模块合并
            self.fusion = FusionModule(config.fusion_strategy, d_enc)
            d_fused = self.fusion.output_dim(d_enc)
        else:
            # 单路模式（M1）：跳过融合，直接将 h_e 送入分类头
            self.fusion = None
            d_fused = d_enc
        self.head = ClassificationHead(d_fused, dropout=0.5)

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        将输入张量编码为固定长度向量。
        输入形状由具体模型决定（onehot: (B,5,L)，kmer: (B,L)，llm: (B,d)）。
        输出：(B, d_enc)
        """
        ...

    def forward(self, x_e: Tensor, x_p: Tensor) -> Tensor:
        """
        前向传播：编码 → 融合 → 分类头。
        返回未经 sigmoid 的 logit，形状 (B,)。
        """
        h_e = self.encode(x_e)                               # (B, d_enc)
        if self.dual_branch:
            h_p = self.encode(x_p)                           # (B, d_enc)
            m   = self.fusion(h_e, h_p)                      # (B, d_fused)
        else:
            m = h_e                                          # 单路：直接送分类头
        return self.head(m)                                  # (B,)
