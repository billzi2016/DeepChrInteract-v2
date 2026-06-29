# DeepChrInteract-v2

基于 PyTorch 的增强子–启动子相互作用（EPI）预测框架，是原始 [DeepChrInteract](https://github.com/lichen-lab/DeepChrInteract)（Keras 1.x）的现代化复现与扩展。

原始代码因 Keras/TensorFlow API 弃用已无法运行，本仓库在保留原始核心设计的基础上，
引入了 14 种编码器架构（包含 Mamba、RWKV、mLSTM、Linear Transformer、iTransformer、MAE 等前沿模型），
并实现了 6 种增强子–启动子融合策略。

## 文档

- 文档源码：`doc/`
- 双语 Sphinx 文档入口：`doc/source/index.rst`
- 本地构建：`make -C doc html`
- 本地生成后的首页：`doc/build/html/index.html`
- GitHub Pages 预期地址：<https://billzi2016.github.io/DeepChrInteract-v2/>

仓库已补充 GitHub Pages 工作流：

- `.github/workflows/docs.yml`

首次启用时，需要在 GitHub 仓库设置中打开 Pages，并将 Source 设为 `GitHub Actions`。

## CI/CD

- 英文版：[`CICD.md`](CICD.md)
- 中文版：[`CICD_CN.md`](CICD_CN.md)

### 文档页面预览

首页：

![Documentation Home](assets/docs/docs-home.png)

English：

![Documentation English](assets/docs/docs-en.png)

中文：

![Documentation Chinese](assets/docs/docs-zh.png)

### 模型文档页面总览

模型总览：

![Model Overview](assets/docs/models/model-overview.png)

CNN 模型：

![CNN Models](assets/docs/models/cnn-models.png)

ResNet 模型：

![ResNet Models](assets/docs/models/resnet-models.png)

BiLSTM 模型：

![BiLSTM Model](assets/docs/models/bilstm-model.png)

mLSTM 模型：

![mLSTM Model](assets/docs/models/mlstm-model.png)

Transformer 模型：

![Transformer Model](assets/docs/models/transformer-model.png)

Linear Transformer 模型：

![Linear Transformer Model](assets/docs/models/linear-transformer-model.png)

iTransformer 模型：

![iTransformer Model](assets/docs/models/itransformer-model.png)

Mamba 模型：

![Mamba Model](assets/docs/models/mamba-model.png)

RWKV 模型：

![RWKV Model](assets/docs/models/rwkv-model.png)

混合模型：

![Hybrid Models](assets/docs/models/hybrid-models.png)

DNA 基础模型：

![DNA Foundation Models](assets/docs/models/dna-foundation-models.png)

MAE 预训练模型：

![MAE Model](assets/docs/models/mae-model.png)

融合策略：

![Fusion Strategies](assets/docs/models/fusion-strategies.png)

## 目录结构

```text
.
├── src/
│   ├── config.py          # 超参数管理（dataclass + argparse）
│   ├── encoders.py        # DNA 序列编码（One-Hot / k-mer / LLM）
│   ├── dataset.py         # PyTorch Dataset（onehot/kmer/llm 三种模式）
│   ├── train.py           # 训练主循环
│   ├── evaluate.py        # 评估系统（AUROC/AUPRC/F1/Accuracy）
│   └── models/
├── scripts/
│   ├── preprocess.py
│   ├── run_experiment.sh
│   └── test_pipeline.py
├── latex/
├── DeepChrInteract-main(old)/
├── PRD.md
└── TASK.md
```

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

## 安装

```bash
pip install -r requirements.txt
pip install mamba-ssm
```

## 数据格式

每个细胞系需要 4 个纯文本文件：

```text
data/raw/{cell_type}/
    seq.anchor1.pos.txt
    seq.anchor2.pos.txt
    seq.anchor1.neg.txt
    seq.anchor2.neg.txt
```

## 快速开始

### 1. 数据预处理

```bash
python scripts/preprocess.py \
    --raw_dir data/raw \
    --cell_type GM12878 \
    --out_dir data
```

### 2. 无数据管道测试

```bash
python scripts/test_pipeline.py
python scripts/test_pipeline.py --quick
```

### 3. 单实验训练

```bash
python -m src.train \
    --model_id M2 \
    --exp_id E03 \
    --encoding_mode onehot \
    --fusion_strategy concat_sub_mul \
    --cell_type GM12878 \
    --seed 0
```

### 4. 五个种子完整实验

```bash
bash scripts/run_experiment.sh E03 M2 GM12878 onehot concat_sub_mul
```

## 输出结构

```text
results/{exp_id}/seed{n}/
    config.json
    best.pt
    last.pt
    history.json
    metrics.json
    roc_curve.png
    pr_curve.png
    summary.json
```

## 与原始代码的主要差异

| 方面 | 原始（Keras 1.x） | v2（PyTorch） |
|------|-----------------|--------------|
| 框架 | Keras + TF 2.3 | PyTorch 2.0+ |
| 中间文件 | PNG（imageio + ImageDataGenerator） | **无**（在线计算） |
| 模型数量 | 4 种 | **14 种** |
| 融合策略 | concat | **6 种** |
| 评估指标 | Accuracy | AUROC + AUPRC + F1 + Accuracy |
| 早停 | ✗ | ✓ |
| 断点续训 | ✗ | ✓ |
| 多 seed 汇总 | ✗ | ✓ |

## 原始实现存档

`DeepChrInteract-main(old)/` 目录保留了原始 Keras 实现供参考。
