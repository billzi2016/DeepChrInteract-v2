# DeepChrInteract-v2：增强子-启动子相互作用预测系统重现与升级

## 背景

本项目是对 **DeepChrInteract**（原作者：本人，代码仓库已 Fork 保留于 `DeepChrInteract-main(old)/`）的完整重现与现代化重写。

原版代码基于 Keras 1.x / TensorFlow 2.x 早期版本，由于以下原因已无法正常运行：

- 使用了大量废弃 API（`fit_generator`、`flow_from_directory` 旧参数等）
- 把 DNA 序列先渲染成 PNG 图片落盘再喂给模型（流程繁琐、I/O 极慢、引入 8-bit 量化误差，且会生成数十 GB 临时文件）
- 无 BiLSTM、Transformer、DNA LLM 等现代序列建模组件
- 缺乏完整评估指标（仅 Accuracy，无 AUROC / AUPRC）
- 代码结构混乱，中英文混杂，无模块化设计

本次重写目标：**保留原始科学思路，以现代 PyTorch 生态全面重实现，并扩展更先进的编码与融合策略，包含 DNA 大语言模型编码器。**

---

## 问题陈述

染色质三维结构中，**增强子（Enhancer）** 与 **启动子（Promoter）** 的远程相互作用（EPI）是基因调控的核心机制之一。基于 DNA 原始序列预测 EPI 是生物信息学的重要任务，可以：

- 规避昂贵的 Hi-C / ChIA-PET 等实验
- 发现未被注释的调控关系
- 辅助疾病相关变异位点解析

**输入**：增强子区域 DNA 序列 + 启动子区域 DNA 序列（均约 10 kb）  
**输出**：二分类——该对 E-P 是否存在相互作用（1 = 有，0 = 无）

---

## 数据格式

原始数据来自公开 Hi-C 数据集处理后的锚点序列，格式为每行一条 DNA 序列的纯文本文件：

```
data/{gene_name}/
  seq.anchor1.pos.txt   # 增强子序列，正样本（有相互作用）
  seq.anchor1.neg2.txt  # 增强子序列，负样本（无相互作用）
  seq.anchor2.pos.txt   # 启动子序列，正样本
  seq.anchor2.neg2.txt  # 启动子序列，负样本
```

字符集：`A C G T N`（N 表示未知碱基）

**序列长度**：默认 10,001 bp，可配置。

---

## 编码方案

### 方案 A：One-Hot 编码（主力方案）

将每个碱基映射为 5 维向量（含 N）：

```
A → [1, 0, 0, 0, 0]
C → [0, 1, 0, 0, 0]
G → [0, 0, 1, 0, 0]
T → [0, 0, 0, 1, 0]
N → [0, 0, 0, 0, 1]
```

输入 shape：`(batch, 5, seq_len)`，类比 1D 音频信号的多通道处理。

**实现**：纯 NumPy 向量化，在 `Dataset.__getitem__` 中动态生成，**完全不写任何中间文件（不存 PNG，不存 npy，直接 txt → tensor）**。

> **关键设计决策**：原版把序列转成 uint8 PNG 落盘（数十 GB），再用 ImageDataGenerator 读回 float，经历两次类型转换损失精度且极慢。v2 完全消除这一流程。

### 方案 B：K-mer Token 编码

将 DNA 序列以滑动窗口切分为 6-mer（ACGT 排列组合共 4096 种 + null），每个 k-mer 映射为整数 token，再接可训练 Embedding 层（dim=128）。

词表大小：4097（含 null）  
实现：`torchtext` / 自定义 `Tokenizer`，兼容原版 6-mer 词表。

### 方案 C：字符级可学习 Embedding（对比基线）

直接将 A/C/G/T/N 映射为 5 个可学习向量（dim=64），无监督预训练或随机初始化均可。

### 方案 D：DNA 大语言模型编码器（新增）

利用在数十亿碱基对上预训练的 DNA LLM 作为特征提取器，提供最丰富的序列上下文表示：

| 模型 | 参数量 | 训练数据 | 上下文长度 | 特点 |
|------|--------|---------|-----------|------|
| **DNABERT** | 86M | 人类参考基因组 | 512 tokens | 6-mer BERT，最广泛使用 |
| **DNABERT-2** | 117M | 135 物种基因组 | 2048 tokens | BPE 分词，FlashAttention，多物种 |
| **Nucleotide Transformer** | 50M–2.5B | 3202 人类基因组 + 850 物种 | 2048 tokens | 最大规模，低数据场景效果优秀 |
| **HyenaDNA** | 1.6M–6.5M | 人类参考基因组 | 1M tokens | 线性复杂度，可处理完整 10kb 序列 |

使用方式：从序列文本直接调用模型的 `encode()` 接口，取 `[CLS]` token 或全局平均池化作为定长向量，再接 projection layer 和分类头。可选 frozen（仅训练分类头）或 fine-tune（全参数微调）模式。

---

## 模型架构

所有模型均为二分类（sigmoid 输出），双路结构（E、P 各自编码后融合）为默认。
共 14 个编码器变体，按技术路线分为 5 组。

---

### A 组：经典 CNN（原版复现）

| 编号 | 名称 | 输入 | 核心结构 |
|------|------|------|---------|
| M1 | CNN 单路 | One-Hot (5, L) | Conv1d×3(64ch,k=24,s=4)+BN+ReLU+MaxPool，×2 stage |
| M2 | CNN 双路 | One-Hot (5, L) | 双份 M1，特征后融合 |
| M3 | k-mer + CNN | Token (L-5) | Embedding(4097→128) + Conv1d×4(64ch,k=32,s=8) |

> M1/M2 复现原版 `model_onehot_cnn_one/two_branch`，修正原版错误（Conv2d→Conv1d）。

---

### B 组：RNN / LSTM 族

| 编号 | 名称 | 核心结构 | 特点 |
|------|------|---------|------|
| M4 | BiLSTM | BiLSTM(hidden=256, layers=2, dropout=0.3) | 双向，捕获前后文依赖 |
| M5 | mLSTM (xLSTM) | 2×双向 mLSTM (d_k=d_v=128，矩阵记忆 C_t) | 矩阵记忆，训练时完全可并行 |

---

### C 组：Transformer 族

| 编号 | 名称 | 复杂度 | 核心结构 |
|------|------|--------|---------|
| M6 | Transformer（标准） | O(L²)→CNN 降至~500 token | CNN 前端降采样 + 4层 Transformer (d=256, nhead=8) + CLS |
| M7 | Linear Transformer | O(Ld²) | ELU+1 核函数近似注意力，**无需降采样** |
| M8 | iTransformer | O(C²L)，C=5 | **反转注意力**：在 5 个核苷酸通道间建模，FFN 处理位置 |

---

### D 组：线性递推 / 状态空间模型

| 编号 | 名称 | 复杂度 | 核心结构 |
|------|------|--------|---------|
| M9 | Mamba | O(L) | 选择性 SSM（输入依赖 A/B/C 矩阵），并行 associative scan |
| M10 | RWKV | O(L) 训练，O(1)/步推理 | Time-mixing（指数衰减）+ Channel-mixing，4层 d=256 |

---

### E 组：混合 + 预训练编码器

| 编号 | 名称 | 核心结构 |
|------|------|---------|
| M11 | CNN + BiLSTM | CNN 特征提取（~500 tokens）→ BiLSTM |
| M12 | CNN + Transformer | CNN 降采样 → 标准 Transformer |
| M13 | DNA LLM（冻结/微调） | DNABERT / DNABERT-2 / Nucleotide Transformer / HyenaDNA → 线性投影 |
| M14 | MAE 预训练 Transformer | 先用遮蔽自编码（掩码率 75%）无监督预训练 M6 的编码器，再在 EPI 标签上微调 |

---

## 多路融合策略

对于双路模型（E 特征向量 `e`，P 特征向量 `p`），提供以下融合方式，可通过配置切换：

| 策略 | 公式 | 说明 |
|------|------|------|
| `concat` | `[e; p]` | 直接拼接，维度翻倍 |
| `add` | `e + p` | 逐元素求和 |
| `subtract` | `e - p` | 方向差异，原版思路 |
| `multiply` | `e ⊙ p` | 逐元素乘积，捕获协同激活 |
| `bilinear` | `e W p^T` | 双线性交互，最强但参数多 |
| `concat+sub+mul` | `[e; p; e-p; e⊙p]` | 综合方案，推荐用于最终模型 |

---

## 工程设计

### 目录结构

```
DeepChrInteract-v2/
├── data/                        # 原始序列数据（不入 git）
├── src/
│   ├── dataset.py               # Dataset、DataLoader、在线 One-Hot
│   ├── encoders.py              # OneHotEncoder、KmerTokenizer、EmbeddingEncoder
│   ├── models/
│   │   ├── cnn.py               # M1/M2 CNN
│   │   ├── bilstm.py            # M4 BiLSTM
│   │   ├── transformer.py       # M5 Transformer
│   │   ├── cnn_bilstm.py        # M6 混合
│   │   ├── cnn_transformer.py   # M7 混合
│   │   ├── llm_encoder.py       # M8 DNA LLM adapter
│   │   ├── mamba.py             # M9 Mamba SSM
│   │   ├── linear_transformer.py # M10 Linear Attention
│   │   ├── itransformer.py      # M11 iTransformer
│   │   ├── rwkv.py              # M12 RWKV
│   │   ├── mlstm.py             # M13 mLSTM (xLSTM)
│   │   └── fusion.py            # 所有融合策略
│   ├── train.py                 # 训练主循环（早停、LR 调度、权重保存）
│   ├── evaluate.py              # AUROC、AUPRC、Accuracy、F1 计算
│   └── config.py                # 所有超参数集中管理（dataclass）
├── scripts/
│   ├── preprocess.py            # 数据预处理（直接输出 .npz，不存 PNG）
│   └── run_experiment.sh        # 批量实验脚本
├── notebooks/
│   └── analysis.ipynb           # 结果可视化
├── DeepChrInteract-main(old)/   # 原版代码存档
└── PRD.md
```

### 核心设计原则

1. **无 PNG 存储**：所有 One-Hot 编码在 `__getitem__` 中动态计算，直接返回 Tensor，不落盘任何中间图片。

2. **统一接口**：所有模型继承 `BaseEPIModel`，统一 `forward(enhancer, promoter) → logit` 签名。

3. **配置驱动**：通过 `config.py`（`dataclass` + `argparse`）控制所有超参，无魔法数字散落代码中。

4. **实验可复现**：固定随机种子，每次实验保存完整配置快照到 `results/` 目录。

5. **评估完整**：训练结束后自动输出 AUROC、AUPRC、Accuracy、F1，并绘制 ROC 曲线与 PR 曲线。

---

## 超参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_len` | 10001 | 序列长度 |
| `batch_size` | 32 | |
| `lr` | 5e-5 | Adam |
| `weight_decay` | 1e-5 | |
| `max_epochs` | 200 | |
| `early_stop_patience` | 15 | 监控 val AUROC |
| `dropout` | 0.5 | |
| `cnn_channels` | [64, 128] | 每阶段 channel 数 |
| `cnn_kernel` | 24 | |
| `lstm_hidden` | 256 | BiLSTM 单向维度 |
| `lstm_layers` | 2 | |
| `d_model` | 256 | Transformer 嵌入维度 |
| `nhead` | 8 | |
| `tf_layers` | 4 | Transformer 层数 |
| `fusion` | `concat+sub+mul` | 融合策略 |
| `encoder` | `onehot` | `onehot` / `kmer` / `embedding` |

---

## 实验矩阵

以下组合将系统性对比，验证各模块贡献：

| 实验 ID | 编码器 | 架构 | 融合 |
|---------|--------|------|------|
| E01 | One-Hot | CNN (单路) | — |
| E02 | One-Hot | CNN (双路) | concat |
| E03 | One-Hot | CNN (双路) | concat+sub+mul |
| E04 | K-mer | CNN (双路) | concat+sub+mul |
| E05 | One-Hot | BiLSTM (双路) | concat+sub+mul |
| E06 | One-Hot | Transformer (双路) | concat+sub+mul |
| E07 | One-Hot | CNN+BiLSTM (双路) | concat+sub+mul |
| E08 | One-Hot | CNN+Transformer (双路) | concat+sub+mul |
| E09 | DNABERT-2 | M13 LLM frozen (双路) | concat+sub+mul |
| E10 | HyenaDNA | M13 LLM fine-tune (双路) | concat+sub+mul |
| E11 | One-Hot | M9 Mamba (双路) | concat+sub+mul |
| E12 | One-Hot | M7 Linear Transformer (双路) | concat+sub+mul |
| E13 | One-Hot | M8 iTransformer (双路) | concat+sub+mul |
| E14 | One-Hot | M10 RWKV (双路) | concat+sub+mul |
| E15 | One-Hot | M5 mLSTM / xLSTM (双路) | concat+sub+mul |
| E16 | One-Hot | M14 MAE 预训练 Transformer (双路) | concat+sub+mul |

---

## 评估指标

- **AUROC**（主要指标，与原版论文对齐）
- **AUPRC**（正负样本不均衡时更可靠）
- **Accuracy**
- **F1 Score**（阈值 0.5）
- 训练曲线（loss / AUROC / AUPRC per epoch）

---

## 与原版的关键差异总结

| 维度 | 原版 (DeepChrInteract) | 本版 (v2) |
|------|----------------------|-----------|
| 框架 | Keras 1.x / TF 2.x | PyTorch 2.x |
| 序列处理 | 存为 PNG 再用 ImageDataGenerator | 在线 One-Hot，纯 Tensor |
| 卷积 | Conv2d 处理序列（不合理） | Conv1d |
| 序列模型 | 无 | BiLSTM、Transformer、Mamba、Linear Transformer、iTransformer、RWKV、mLSTM |
| 融合策略 | 仅 concat | concat / sub / mul / bilinear / 组合 |
| 评估 | 仅 Accuracy | AUROC + AUPRC + F1 |
| 代码质量 | 单文件、魔法数字、中英混杂 | 模块化、配置驱动、类型注解 |
| 数据预处理 | imageio + skimage 存图 | numpy 向量化，直接 .npz |

---

## 依赖

```
torch >= 2.0
numpy
scikit-learn
matplotlib
tqdm
```

可选（K-mer 编码）：
```
torchtext
```
