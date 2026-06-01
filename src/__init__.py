"""
src — DeepChrInteract-v2 核心包

子模块说明：
  config   — 超参数管理（Config dataclass + argparse）
  encoders — DNA 序列编码工具（OneHot / k-mer / LLM）
  dataset  — PyTorch Dataset（EPIDataset + DummyEPIDataset）
  models   — 14 个编码器模型（M1–M14，Groups A–E）
  train    — 训练主循环（Adam + CosineWarmRestart + 早停）
  evaluate — 评估系统（AUROC / AUPRC / F1 / Accuracy）
"""
