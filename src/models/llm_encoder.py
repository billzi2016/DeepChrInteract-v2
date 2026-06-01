"""
llm_encoder.py — Group E：DNA 基础模型编码器（M13）

统一封装四种 DNA 大语言模型：
  dnabert    — DNABERT（Ji et al., Bioinformatics 2021），6-mer tokenize + BERT
  dnabert2   — DNABERT-2（Zhou et al., ICLR 2024），BPE tokenize，多物种预训练
  nt         — Nucleotide Transformer 500M（Dalla-Torre et al., Nature Methods 2024）
  hyenadna   — HyenaDNA（Nguyen et al., NeurIPS 2023），单碱基 token，O(L) Hyena 算子

两种使用模式：
  frozen   — LLM 权重冻结，仅训练线性投影层和分类头（lr=5e-5）
  finetune — LLM 以 lr=5e-6 微调，分类头以 lr=5e-5 微调

前向传播流程：
  序列字符串 → HuggingFace tokenizer → LLM → 均值池化 → 线性投影 → 融合 → 分类头

注：M13 的输入是原始序列字符串（与其他模型接收 tensor 不同），
    由 EPIDataset 的 mode='llm' 分支在 collate 时处理，
    或通过 LLMEncoder.encode_dataset() 离线预计算嵌入后转为 mode='llm' tensor。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from .base import BaseEPIModel

# HuggingFace 模型 ID 映射
_HF_IDS = {
    "dnabert":  "zhihan1996/DNA_bert_6",
    "dnabert2": "zhihan1996/DNABERT-2-117M",
    "nt":       "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "hyenadna": "LongSafari/hyenadna-small-32k-seqlen-hf",
}

# 各模型默认输出维度（隐层大小）
_LLM_DIMS = {
    "dnabert":  768,
    "dnabert2": 768,
    "nt":       1024,
    "hyenadna": 256,
}


class M13_LLMEncoder(BaseEPIModel):
    """
    DNA 基础模型编码器（M13）。

    当 config.llm_frozen=True 时，LLM 参数被冻结，仅投影层参与训练；
    当 config.llm_frozen=False 时，需在优化器中对 LLM 参数使用较小学习率
    （train.py 中已处理参数分组）。

    forward() 接受预计算的 LLM 嵌入向量（由 LLMEncoder.encode_dataset() 生成），
    形状 (B, d_llm)，经线性投影后送入融合模块。
    这使得 M13 能与其他模型使用完全相同的训练循环。
    """

    def __init__(self, config: Config):
        backbone = config.llm_backbone
        if backbone not in _HF_IDS:
            raise ValueError(f"llm_backbone 必须是 {list(_HF_IDS.keys())} 之一")

        d_llm = _LLM_DIMS[backbone]
        D_ENC = 256                     # 统一投影到 D_ENC 维
        super().__init__(config, D_ENC, dual_branch=True)

        self.backbone = backbone
        self.frozen = config.llm_frozen
        self._tokenizer = None
        self._llm = None

        # 线性投影：d_llm → D_ENC
        self.proj = nn.Sequential(
            nn.Linear(d_llm, D_ENC),
            nn.LayerNorm(D_ENC),
            nn.GELU(),
        )

    def _load_llm(self) -> None:
        """懒加载 LLM（仅在需要在线推理时触发）。"""
        if self._llm is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        hf_id = _HF_IDS[self.backbone]
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self._llm = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        if self.frozen:
            for p in self._llm.parameters():
                p.requires_grad = False
        self._llm.to(next(self.parameters()).device)

    def encode(self, x: Tensor) -> Tensor:
        """
        x: (B, d_llm) — 预计算的 LLM 嵌入向量
        → (B, D_ENC)
        """
        return self.proj(x)

    def encode_from_sequences(self, seqs: list, max_length: int = 512) -> Tensor:
        """
        在线编码（直接输入序列字符串）。
        通常不在训练循环中使用；建议提前运行 LLMEncoder.encode_dataset()。
        """
        self._load_llm()
        enc = self._tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad() if self.frozen else torch.enable_grad():
            out = self._llm(**enc)

        embeds = out.last_hidden_state.mean(dim=1)   # (B, d_llm)
        return self.proj(embeds)                      # (B, D_ENC)

    def get_param_groups(self, head_lr: float, llm_lr: float) -> list:
        """
        返回参数分组，供 train.py 构建优化器时使用。
        frozen 模式：LLM 参数不包含在优化器中。
        finetune 模式：LLM 参数使用较小 llm_lr，其余参数使用 head_lr。
        """
        proj_head_params = list(self.proj.parameters()) + \
                           list(self.fusion.parameters()) + \
                           list(self.head.parameters())

        if self.frozen or self._llm is None:
            return [{"params": proj_head_params, "lr": head_lr}]
        else:
            return [
                {"params": proj_head_params,        "lr": head_lr},
                {"params": self._llm.parameters(),  "lr": llm_lr},
            ]
