"""
models/__init__.py — 模型注册表

通过 build_model(model_id, config) 统一构建所有 14 个编码器。
"""

from __future__ import annotations

from typing import Dict, Type

from ..config import Config
from .base import BaseEPIModel
from .bilstm import M4_BiLSTM
from .cnn import M1_CNN_SingleBranch, M2_CNN_DualBranch, M3_Kmer_CNN
from .hybrid import M11_CNN_BiLSTM, M12_CNN_Transformer
from .llm_encoder import M13_LLMEncoder
from .mae import M14_MAE_Transformer
from .mamba import M9_Mamba
from .mlstm import M5_mLSTM
from .rwkv import M10_RWKV
from .transformer import M6_Transformer, M7_LinearTransformer, M8_iTransformer

# 全局模型注册表：model_id → 类
MODEL_REGISTRY: Dict[str, Type[BaseEPIModel]] = {
    "M1":  M1_CNN_SingleBranch,
    "M2":  M2_CNN_DualBranch,
    "M3":  M3_Kmer_CNN,
    "M4":  M4_BiLSTM,
    "M5":  M5_mLSTM,
    "M6":  M6_Transformer,
    "M7":  M7_LinearTransformer,
    "M8":  M8_iTransformer,
    "M9":  M9_Mamba,
    "M10": M10_RWKV,
    "M11": M11_CNN_BiLSTM,
    "M12": M12_CNN_Transformer,
    "M13": M13_LLMEncoder,
    "M14": M14_MAE_Transformer,
}


def build_model(model_id: str, config: Config) -> BaseEPIModel:
    """根据 model_id 和配置实例化对应编码器模型。"""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型 '{model_id}'，可选：{list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id](config)


__all__ = ["build_model", "MODEL_REGISTRY", "BaseEPIModel"]
