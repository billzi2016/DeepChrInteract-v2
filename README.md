# DeepChrInteract-v2

基于 PyTorch 的增强子–启动子相互作用（EPI）预测框架，是原始 [DeepChrInteract](https://github.com/lichen-lab/DeepChrInteract)（Keras 1.x）的现代化复现与扩展。

原始代码因 Keras/TensorFlow API 弃用已无法运行，本仓库在保留原始核心设计的基础上，
引入了 14 种编码器架构（包含 Mamba、RWKV、mLSTM、Linear Transformer、iTransformer、MAE 等前沿模型），
并实现了 6 种增强子–启动子融合策略。

---

## 目录结构

```
.
├── src/
│   ├── config.py          # 超参数管理（dataclass + argparse）
│   ├── encoders.py        # DNA 序列编码（One-Hot / k-mer / LLM）
│   ├── dataset.py         # PyTorch Dataset（onehot/kmer/llm 三种模式）
│   ├── train.py           # 训练主循环
│   ├── evaluate.py        # 评估系统（AUROC/AUPRC/F1/Accuracy）
│   └── models/
│       ├── base.py        # 基类 + 分类头
│       ├── fusion.py      # 6 种融合策略
│       ├── cnn.py         # M1, M2, M3
│       ├── bilstm.py      # M4
│       ├── mlstm.py       # M5（xLSTM / Bio-xLSTM）
│       ├── transformer.py # M6, M7（Linear Transformer）, M8（iTransformer）
│       ├── mamba.py       # M9
│       ├── rwkv.py        # M10
│       ├── hybrid.py      # M11（CNN+BiLSTM）, M12（CNN+Transformer）
│       ├── llm_encoder.py # M13（DNABERT / DNABERT-2 / NT / HyenaDNA）
│       └── mae.py         # M14（MAE 预训练）
├── scripts/
│   ├── preprocess.py      # 原始 .txt → train/val/test.npz
│   ├── run_experiment.sh  # 单实验 5-seed 完整流程
│   └── test_pipeline.py   # 管道完整性测试（无需真实数据）
├── latex/                 # 技术文档（LaTeX）
├── DeepChrInteract-main(old)/  # 原始 Keras 代码（存档）
├── PRD.md                 # 需求文档
└── TASK.md                # 任务进度看板
```

---

## 模型列表（M1–M14）

| 组别 | ID  | 模型 | 编码方式 |
|------|-----|------|---------|
| A | M1  | CNN 单路（Baseline 复现） | One-Hot |
| A | M2  | CNN 双路（原始论文复现） | One-Hot |
| A | M3  | k-mer Embedding + CNN | k-mer |
| B | M4  | 双向 LSTM | One-Hot |
| B | M5  | mLSTM（xLSTM / Bio-xLSTM） | One-Hot |
| C | M6  | 标准 Transformer | One-Hot |
| C | M7  | Linear Transformer（O(Ld²)） | One-Hot |
| C | M8  | iTransformer（通道维 Attention） | One-Hot |
| D | M9  | Mamba（选择性 SSM） | One-Hot |
| D | M10 | RWKV（指数衰减线性递推） | One-Hot |
| E | M11 | CNN + BiLSTM | One-Hot |
| E | M12 | CNN + Transformer | One-Hot |
| E | M13 | DNA LLM（DNABERT-2 / NT / HyenaDNA / DNABERT） | LLM 嵌入 |
| E | M14 | MAE 预训练 Transformer | One-Hot |

### 融合策略（6 种）

| 策略 | 公式 | 输出维度 |
|------|------|---------|
| concat | [h_e; h_p] | 2d |
| add | h_e + h_p | d |
| subtract | h_e − h_p | d |
| multiply | h_e ⊙ h_p | d |
| bilinear | h_e^T W h_p | 1 |
| **concat_sub_mul** *(默认)* | [h_e; h_p; h_e−h_p; h_e⊙h_p] | 4d |

---

## 安装

```bash
pip install -r requirements.txt

# 可选（需要 CUDA + 特殊编译）：
pip install mamba-ssm
```

---

## 数据格式

每个细胞系需要 4 个纯文本文件（每行一条 DNA 序列）：

```
data/raw/{cell_type}/
    seq.anchor1.pos.txt   # 正样本增强子序列
    seq.anchor2.pos.txt   # 正样本启动子序列
    seq.anchor1.neg.txt   # 负样本增强子序列
    seq.anchor2.neg.txt   # 负样本启动子序列
```

---

## 快速开始

### 1. 数据预处理（有真实数据时）

```bash
python scripts/preprocess.py \
    --raw_dir data/raw \
    --cell_type GM12878 \
    --out_dir data
```

### 2. 无数据管道测试

```bash
# 测试全部 14 个模型 + 训练循环 + 评估（随机张量，无需数据）
python scripts/test_pipeline.py

# 仅快速测试部分模型
python scripts/test_pipeline.py --quick
```

### 3. 单实验训练

```bash
# M2（CNN 双路），one-hot，concat_sub_mul
python -m src.train \
    --model_id M2 \
    --exp_id E03 \
    --encoding_mode onehot \
    --fusion_strategy concat_sub_mul \
    --cell_type GM12878 \
    --seed 0
```

### 4. 完整 5-seed 实验（含评估）

```bash
bash scripts/run_experiment.sh E03 M2 GM12878 onehot concat_sub_mul
```

### 5. M14 MAE 预训练 + 微调

```bash
# 第一阶段：MAE 预训练
python -m src.train --model_id M14 --exp_id E16 --pretrain

# 第二阶段：加载预训练权重微调（在 train.py 中通过 load_pretrained() 实现）
python -m src.train --model_id M14 --exp_id E16
```

### 6. M13 LLM 编码器（冻结模式）

```bash
# 先离线生成 LLM 嵌入（只需运行一次）
python -c "
from src.encoders import LLMEncoder
import numpy as np
enc = LLMEncoder('dnabert2')
# seqs_e, seqs_p 从 npz 读取后传入
enc.encode_dataset(seqs_e, seqs_p, out_dir='data/GM12878/llm_dnabert2')
"

# 再训练
python -m src.train \
    --model_id M13 \
    --encoding_mode llm \
    --llm_backbone dnabert2 \
    --llm_frozen \
    --exp_id E09
```

---

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | M1 | 模型编号 M1–M14 |
| `--encoding_mode` | onehot | onehot / kmer / llm |
| `--fusion_strategy` | concat_sub_mul | 见上方融合策略表 |
| `--cell_type` | GM12878 | 细胞系名称 |
| `--batch_size` | 32 | 训练 batch 大小 |
| `--lr` | 5e-5 | Adam 学习率 |
| `--max_epochs` | 100 | 最大训练轮数 |
| `--patience` | 15 | 早停 patience |
| `--seed` | 0 | 随机种子 |
| `--dummy` | False | 使用随机张量（无需真实数据）|
| `--pretrain` | False | M14 MAE 预训练模式 |
| `--resume` |  | 断点续训 checkpoint 路径 |

---

## 输出结构

```
results/{exp_id}/seed{n}/
    config.json       # 实验配置快照
    best.pt           # 最优 checkpoint（val AUROC）
    last.pt           # 最新 checkpoint
    history.json      # 每 epoch 的 loss / AUROC
    metrics.json      # 测试集指标（AUROC/AUPRC/F1/Accuracy）
    roc_curve.png     # ROC 曲线
    pr_curve.png      # Precision-Recall 曲线
    summary.json      # 多 seed 均值 ± std（在 evaluate 后生成）
```

---

## 与原始代码的主要差异

| 方面 | 原始（Keras 1.x） | v2（PyTorch） |
|------|-----------------|--------------|
| 框架 | Keras + TF 2.3 | PyTorch 2.0+ |
| 中间文件 | PNG（imageio + ImageDataGenerator） | **无**（在线计算） |
| 模型数量 | 4 种 | **14 种**（含 SSM/LLM/MAE） |
| 融合策略 | concat | **6 种** |
| 评估指标 | Accuracy | AUROC + AUPRC + F1 + Accuracy |
| 早停 | ✗ | ✓（patience=15，val AUROC） |
| 断点续训 | ✗ | ✓ |
| 多 seed CI | ✗ | ✓（5 seed 均值 ± std） |

---

## 论文

技术细节参见 `latex/main.pdf`，内容涵盖：
- 数据编码方法（One-Hot / k-mer / LLM）
- 全部 14 个编码器架构（含公式推导）
- 实验设计（E01–E16 + 融合消融 F01–F06）
- 相关工作综述（Mamba、RWKV、xLSTM、iTransformer、MAE、DNA 基础模型）

---

## 原始论文存档

`DeepChrInteract-main(old)/` 目录保留了原始 Keras 实现供参考。
作者为本仓库原始作者，已获授权在此存档。
