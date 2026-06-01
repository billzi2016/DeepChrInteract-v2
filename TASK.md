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

- [x] 创建目录结构
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
    test_pipeline.py
  notebooks/
  results/
  data/
  ```
- [x] `requirements.txt` — 固定所有依赖版本
  - [x] `torch >= 2.0`
  - [x] `numpy`, `scikit-learn`, `matplotlib`, `tqdm`
  - [x] `transformers`（DNABERT / NT）
  - [x] 可选：`mamba-ssm`（CUDA 环境）
- [x] `src/config.py` — 全局超参数管理
  - [x] `dataclass` 定义所有超参（lr、batch、dropout、d_model 等）
  - [x] `argparse` 覆盖接口
  - [x] 保存配置快照到 `results/{exp_id}/seed{n}/config.json`
  - [x] 固定随机种子工具函数 `set_seed(seed)`

---

## Phase 2：数据管道

- [x] `scripts/preprocess.py` — 数据预处理
  - [x] 读取四个 `.txt` 文件（anchor1/2 × pos/neg）
  - [x] 对齐正负样本长度
  - [x] Shuffle + 分层 train/val/test 划分（80/10/10）
  - [x] 保存为 `data/{cell_type}/train|val|test.npz`（序列字符串 + 标签）
  - [x] **不生成任何 PNG / 中间图片文件**

- [x] `src/dataset.py` — PyTorch Dataset
  - [x] `EPIDataset` 类，`__getitem__` 返回 `(x_e, x_p, label)`
  - [x] 支持三种模式：`onehot` / `kmer` / `llm`
  - [x] One-Hot：纯 NumPy 向量化，在线动态生成，不落盘
  - [x] k-mer：滑窗 6-mer → integer token
  - [x] LLM：预计算好的 embedding 直接读 `.npy`
  - [x] `DummyEPIDataset`（无数据时验证管道）
  - [x] `get_dataloader()` 工厂函数（train/val/test，worker 数可配）

- [x] `src/encoders.py` — 编码器工具
  - [x] `OneHotEncoder` — `(seq_str,) → Tensor(5, L)`
  - [x] `KmerTokenizer` — 6-mer 词表构建 + `encode(seq)` → `LongTensor(L-5)`
  - [x] `LLMEncoder` — 批量调用 HuggingFace 模型，输出 `.npy` 缓存文件

---

## Phase 3：模型实现

> 所有模型继承 `BaseEPIModel`，统一签名 `forward(x_e, x_p) → logit`

- [x] `src/models/base.py` — 基类 + 分类头 MLP
  - [x] `BaseEPIModel(nn.Module)`：定义 `encode()` + `fuse()` + `classify()`
  - [x] 分类头：`Linear(d_out, 256) → ReLU → Dropout → Linear(256, 1)`

- [x] `src/models/fusion.py` — 6 种融合策略
  - [x] `FusionModule(strategy)` 支持：`concat` / `add` / `subtract` / `multiply` / `bilinear` / `concat_sub_mul`
  - [x] 默认：`concat_sub_mul`

### Group A：经典 CNN（复现）

- [x] `src/models/cnn.py`
  - [x] `M1_CNN_SingleBranch` — 单路 CNN，复现 `model_onehot_cnn_one_branch`
  - [x] `M2_CNN_DualBranch` — 双路 CNN，复现 `model_onehot_cnn_two_branch`
  - [x] `M3_Kmer_CNN` — Embedding(4097, 128) + Conv1d × 4 + fusion

### Group B：RNN / LSTM 族

- [x] `src/models/bilstm.py`
  - [x] `M4_BiLSTM` — 2 层双向 LSTM（hidden=256，dropout=0.3）+ fusion

- [x] `src/models/mlstm.py`
  - [x] `M5_mLSTM` — mLSTM cell（矩阵记忆 C_t，covariance update，log-space 稳定）
  - [x] 双向包装（前向 + 后向），2 层 stack，d_k = d_v = 128

### Group C：Transformer 族

- [x] `src/models/transformer.py`
  - [x] `M6_Transformer` — CNN 前端（stride=16）+ 4 层 TransformerEncoder + CLS token
  - [x] `M7_LinearTransformer` — ELU+1 核函数 Linear Attention，O(Ld²)
  - [x] `M8_iTransformer` — 在 C=5 通道维做 attention，O(C²L)

### Group D：线性递推 / SSM

- [x] `src/models/mamba.py`
  - [x] `M9_Mamba` — 优先使用 `mamba-ssm` 库，回退到纯 PyTorch fallback

- [x] `src/models/rwkv.py`
  - [x] `M10_RWKV` — Time-mixing（log-space 稳定扫描）+ Channel-mixing，4 层 d=256

### Group E：混合 + 预训练

- [x] `src/models/hybrid.py`
  - [x] `M11_CNN_BiLSTM` — CNN 前端 → BiLSTM
  - [x] `M12_CNN_Transformer` — CNN 前端 → Transformer

- [x] `src/models/llm_encoder.py`
  - [x] `M13_LLMEncoder` — 支持 DNABERT / DNABERT-2 / NT / HyenaDNA
  - [x] frozen 模式 + fine-tune 模式（参数分组不同 lr）

- [x] `src/models/mae.py`
  - [x] `M14_MAE_Transformer` — MAE 预训练（75% 遮蔽，MSE 重建）+ 微调

- [x] `src/models/__init__.py` — `MODEL_REGISTRY` + `build_model()` 工厂

### 测试验证

- [x] `scripts/test_pipeline.py` — 全 16 个测试用例，14 个模型前向/反向，训练+评估循环
- [x] **全部 16/16 通过**（MPS 设备，PyTorch 2.x）

---

## Phase 4：训练系统

- [x] `src/train.py` — 训练主循环
  - [x] 单函数入口 `train(model, config, train_loader, val_loader)`
  - [x] 优化器：Adam（lr=5e-5，weight_decay=1e-5），M13 支持参数分组不同 lr
  - [x] LR 调度：CosineAnnealingWarmRestarts（T_0=50）
  - [x] 损失：BCEWithLogitsLoss
  - [x] 梯度裁剪：max_norm=1.0
  - [x] 早停：patience=15，监控 val AUROC
  - [x] 每 epoch 打印：loss / AUROC
  - [x] 保存最优 checkpoint（`results/{exp_id}/seed{n}/best.pt`）
  - [x] 保存完整训练曲线（`history.json`）
  - [x] 断点续训（`--resume`）
  - [x] GPU 自动检测（cuda → mps → cpu）
  - [x] `--dummy` 标志（随机张量，无需真实数据）

- [x] MAE 预训练入口（`--pretrain` 参数，M14 专用）

---

## Phase 5：评估系统

- [x] `src/evaluate.py`
  - [x] `evaluate(model, test_loader)` → 返回 dict
  - [x] 指标：AUROC、AUPRC、Accuracy、F1（阈值 0.5）
  - [x] 多 seed 均值 ± std 汇总（`summarize_seeds()`）
  - [x] 绘制 ROC 曲线（`roc_curve.png`）
  - [x] 绘制 PR 曲线（`pr_curve.png`）
  - [x] 输出结果 JSON（`results/{exp_id}/seed{n}/metrics.json`）

---

## Phase 6：实验运行

> 每个实验 5 个 seed（0-4），seed 固定。需要真实数据后执行。

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
