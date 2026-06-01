# DeepChrInteract-v2 — Task Board

> 状态：`[ ]` 待做 · `[x]` 完成 · `[-]` 进行中 · `[~]` 暂缓

---

## Phase 0：文档与规划

- [x] 阅读并理解老代码（`DeepChrInteract-main(old)/`）
- [x] 归档老代码进 git
- [x] 编写 PRD.md（需求文档，含 M1–M14 编号体系）
- [x] 编写 LaTeX 技术文档（18 页，含引用）
  - [x] `01_introduction.tex`
  - [x] `02_related_work.tex`（含 Mamba/RWKV/xLSTM/MAE/iTransformer）
  - [x] `03_data_encoding.tex`
  - [x] `04_methods.tex`（M1–M14 + 融合策略 + 分类头）
  - [x] `05_experiments.tex`（E01–E16 实验矩阵）
  - [x] `06_conclusion.tex`
  - [x] `references.bib`（20+ 条验证引用）
  - [x] 编译为 `main.pdf`，无 Overfull 警告
- [x] 编写 TASK.md（本文件）

---

## Phase 1：项目骨架

- [ ] 创建目录结构
  ```
  src/
    dataset.py
    encoders.py
    models/
    train.py
    evaluate.py
    config.py
  scripts/
    preprocess.py
    run_experiment.sh
  notebooks/
    analysis.ipynb
  ```
- [ ] `requirements.txt` — 固定所有依赖版本
  - [ ] `torch >= 2.0`
  - [ ] `numpy`, `scikit-learn`, `matplotlib`, `tqdm`
  - [ ] `transformers`（DNABERT / NT）
  - [ ] `mamba-ssm`（Mamba 官方 CUDA kernel）
  - [ ] 可选：`rwkv`、`flash-attn`
- [ ] `src/config.py` — 全局超参数管理
  - [ ] `dataclass` 定义所有超参（lr、batch、dropout、d_model 等）
  - [ ] `argparse` 覆盖接口
  - [ ] 保存配置快照到 `results/{exp_id}/config.json`
  - [ ] 固定随机种子工具函数 `set_seed(seed)`

---

## Phase 2：数据管道

- [ ] `scripts/preprocess.py` — 数据预处理
  - [ ] 读取四个 `.txt` 文件（anchor1/2 × pos/neg）
  - [ ] 对齐正负样本长度
  - [ ] Shuffle + 分层 train/val/test 划分（80/10/10）
  - [ ] 保存为 `data/{cell_type}/processed.npz`（序列字符串 + 标签）
  - [ ] **不生成任何 PNG / 中间图片文件**

- [ ] `src/dataset.py` — PyTorch Dataset
  - [ ] `EPIDataset` 类，`__getitem__` 返回 `(x_e, x_p, label)`
  - [ ] 支持三种模式：`onehot` / `kmer` / `llm`
  - [ ] One-Hot：纯 NumPy 向量化，在线动态生成，不落盘
  - [ ] k-mer：滑窗 6-mer → integer token，`pad_sequences`
  - [ ] LLM：预计算好的 embedding 直接读 `.npy`
  - [ ] `get_dataloader()` 工厂函数（train/val/test，worker 数可配）

- [ ] `src/encoders.py` — 编码器工具
  - [ ] `OneHotEncoder` — `(seq_str,) → Tensor(5, L)`
  - [ ] `KmerTokenizer` — 6-mer 词表构建 + `encode(seq)` → `LongTensor(L-5)`
  - [ ] `LLMEncoder` — 批量调用 HuggingFace 模型，输出 `.npy` 缓存文件

---

## Phase 3：模型实现

> 所有模型继承 `BaseEPIModel`，统一签名 `forward(x_e, x_p) → logit`

- [ ] `src/models/base.py` — 基类 + 分类头 MLP
  - [ ] `BaseEPIModel(nn.Module)`：定义 `encode()` + `fuse()` + `classify()`
  - [ ] 分类头：`Linear(d_out, 256) → ReLU → Linear(256, 1) → Sigmoid`

- [ ] `src/models/fusion.py` — 6 种融合策略
  - [ ] `FusionModule(strategy)` 支持：`concat` / `add` / `subtract` / `multiply` / `bilinear` / `concat_sub_mul`
  - [ ] 默认：`concat_sub_mul`

### Group A：经典 CNN（复现）

- [ ] `src/models/cnn.py`
  - [ ] `M1_CNN_SingleBranch` — Conv1d × 3 (64ch) + Conv1d × 3 (128ch) + pool + FC
  - [ ] `M2_CNN_DualBranch` — 双份 M1 编码器 + fusion
  - [ ] `M3_Kmer_CNN` — Embedding(4097, 128) + Conv1d × 4 (64ch, k=32, s=8) + fusion

### Group B：RNN / LSTM 族

- [ ] `src/models/bilstm.py`
  - [ ] `M4_BiLSTM` — 2 层双向 LSTM（hidden=256，dropout=0.3）+ fusion

- [ ] `src/models/mlstm.py`
  - [ ] `M5_mLSTM` — mLSTM cell（矩阵记忆 C_t，covariance update）
  - [ ] 双向包装（前向 + 后向）
  - [ ] 2 层 stack，d_k = d_v = 128
  - [ ] 参考 Bio-xLSTM 实现

### Group C：Transformer 族

- [ ] `src/models/transformer.py`
  - [ ] `M6_Transformer` — CNN 前端（stride=16，降至 ~625 tokens）+ sinusoidal PE + 4层 TransformerEncoder（d=256, nhead=8）+ CLS token
  - [ ] `M7_LinearTransformer` — ELU+1 核函数 attention，无需降采样，4层（d=256, nhead=8）
  - [ ] `M8_iTransformer` — 在 C=5 通道维做 attention，FFN 按位置处理；无需降采样

### Group D：线性递推 / SSM

- [ ] `src/models/mamba.py`
  - [ ] `M9_Mamba` — 4 × Mamba Block（d_model=256），使用 `mamba-ssm` 库
  - [ ] MeanPool 聚合 + fusion

- [ ] `src/models/rwkv.py`
  - [ ] `M10_RWKV` — Time-mixing（指数衰减 w，parallel cumsum）+ Channel-mixing，4 层 d=256
  - [ ] MeanPool + fusion

### Group E：混合 + 预训练

- [ ] `src/models/hybrid.py`
  - [ ] `M11_CNN_BiLSTM` — M1 CNN 前端（截断至 ~500 tokens）→ M4 BiLSTM + fusion
  - [ ] `M12_CNN_Transformer` — M1 CNN 前端 → M6 Transformer + fusion

- [ ] `src/models/llm_encoder.py`
  - [ ] `M13_LLMEncoder` — 统一 adapter，支持：
    - [ ] DNABERT（6-mer tokenize → `BertModel` → CLS）
    - [ ] DNABERT-2（BPE tokenize → CLS）
    - [ ] Nucleotide Transformer（500M 变体，mean pool）
    - [ ] HyenaDNA（single-nucleotide token，mean pool）
  - [ ] frozen 模式：仅训练 projection + head
  - [ ] fine-tune 模式：LLM lr=5e-6，head lr=5e-5

- [ ] `src/models/mae.py`
  - [ ] `M14_MAE_Transformer` — MAE 预训练阶段
    - [ ] Masking：随机遮蔽 75% nucleotide tokens
    - [ ] Encoder（M6 架构）+ 轻量 Decoder（2 层 Transformer）
    - [ ] 重建目标：原始 one-hot 向量（MSE loss）
    - [ ] 预训练入口：`pretrain_mae(dataset, encoder, decoder, epochs)`
  - [ ] fine-tune 阶段：丢弃 decoder，encoder 接 fusion + head

---

## Phase 4：训练系统

- [ ] `src/train.py` — 训练主循环
  - [ ] 单函数入口 `train(model, config, train_loader, val_loader)`
  - [ ] 优化器：Adam（lr=5e-5，weight_decay=1e-5）
  - [ ] LR 调度：CosineAnnealingWarmRestarts（T_0=50）
  - [ ] 损失：BCEWithLogitsLoss
  - [ ] 早停：patience=15，监控 val AUROC
  - [ ] 每 epoch 打印：loss / AUROC / AUPRC
  - [ ] 保存最优 checkpoint（`results/{exp_id}/best.pt`）
  - [ ] 保存完整训练曲线（loss / AUROC per epoch → `history.json`）
  - [ ] 支持断点续训（`--resume results/{exp_id}/last.pt`）
  - [ ] GPU 自动检测（cuda → mps → cpu）

- [ ] MAE 预训练入口（`--pretrain` 参数）
  - [ ] 无监督预训练 M14 编码器
  - [ ] 保存预训练权重，供 fine-tune 加载

---

## Phase 5：评估系统

- [ ] `src/evaluate.py`
  - [ ] `evaluate(model, test_loader)` → 返回 dict
  - [ ] 指标：AUROC、AUPRC、Accuracy、F1（阈值 0.5）
  - [ ] 95% CI（5 次独立 seed 的均值 ± std）
  - [ ] 绘制 ROC 曲线（`roc_curve.png`）
  - [ ] 绘制 PR 曲线（`pr_curve.png`）
  - [ ] 输出结果 JSON（`results/{exp_id}/metrics.json`）

---

## Phase 6：实验运行

> 每个实验 5 个 seed（0-4），seed 固定

- [ ] **E01** — M1 CNN 单路，one-hot，concat
- [ ] **E02** — M2 CNN 双路，one-hot，concat
- [ ] **E03** — M2 CNN 双路，one-hot，concat_sub_mul
- [ ] **E04** — M3 k-mer + CNN，双路，concat_sub_mul
- [ ] **E05** — M4 BiLSTM，双路，concat_sub_mul
- [ ] **E06** — M6 标准 Transformer，双路，concat_sub_mul
- [ ] **E07** — M11 CNN+BiLSTM，双路，concat_sub_mul
- [ ] **E08** — M12 CNN+Transformer，双路，concat_sub_mul
- [ ] **E09** — M13 DNABERT-2 frozen，双路，concat_sub_mul
- [ ] **E10** — M13 HyenaDNA fine-tune，双路，concat_sub_mul
- [ ] **E11** — M9 Mamba，双路，concat_sub_mul
- [ ] **E12** — M7 Linear Transformer，双路，concat_sub_mul
- [ ] **E13** — M8 iTransformer，双路，concat_sub_mul
- [ ] **E14** — M10 RWKV，双路，concat_sub_mul
- [ ] **E15** — M5 mLSTM，双路，concat_sub_mul
- [ ] **E16** — M14 MAE 预训练 → fine-tune，双路，concat_sub_mul

### 融合策略消融（基于 E03 基础模型）

- [ ] **F01** — concat
- [ ] **F02** — add
- [ ] **F03** — subtract
- [ ] **F04** — multiply
- [ ] **F05** — bilinear
- [ ] **F06** — concat_sub_mul（默认）

### 跨细胞系泛化

- [ ] 取最优模型，在其他细胞系上零样本测试（不重新训练）

---

## Phase 7：可视化与分析

- [ ] `notebooks/analysis.ipynb`
  - [ ] 读取所有 `metrics.json`，汇总为结果表格
  - [ ] AUROC / AUPRC 柱状图（E01–E16 对比）
  - [ ] 融合策略消融图
  - [ ] 训练曲线可视化（loss / AUROC per epoch）
  - [ ] ROC / PR 曲线多模型对比图

---

## Phase 8：LaTeX 填写实验结果

- [ ] 用实际跑出的数字填充 `05_experiments.tex` 中的 `---` 占位符
- [ ] 更新 `04_methods.tex` 的超参（如有调整）
- [ ] 重新编译 `main.pdf`

---

## 优先级排序（建议执行顺序）

```
Phase 1 → Phase 2 → Phase 3（A组先，验证 pipeline）
→ Phase 4 → Phase 5
→ Phase 3（B/C/D/E 组）
→ Phase 6
→ Phase 7 → Phase 8
```
