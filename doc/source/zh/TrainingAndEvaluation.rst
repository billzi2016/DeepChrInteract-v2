训练与评估
==========

切换语言: :doc:`../en/TrainingAndEvaluation`

训练入口
+++++++++++++++

当前训练主入口是 ``python -m src.train``。训练脚本围绕配置对象、数据集构建、
模型实例化、基于验证集的 checkpoint 保存以及 JSON 日志记录组成完整实验闭环。

训练系统核心行为
+++++++++++++++++++++++++++++++++++++++

训练系统包括以下部分：

- 在每个实验目录保存 ``config.json``；
- 自动设备选择，顺序为 ``cuda -> mps -> cpu``；
- 支持在没有真实基因组数据时使用 dummy 模式验证完整管道；
- 支持 k-mer token 化和 LLM embedding 读取；
- 使用 Adam 优化器；
- DNA LLM 微调时支持参数分组；
- 使用 cosine annealing warm restarts 调度；
- 使用 ``max_norm=1.0`` 的梯度裁剪；
- 以验证集 AUROC 为 early stopping 指标；
- 保存 ``best.pt`` 和 ``last.pt``。

输入管道
+++++++++++++++

数据层围绕 enhancer-promoter 配对输入设计：

- ``EPIDataset`` 从 NPZ 中读取 ``seqs_e``、``seqs_p``、``labels``；
- ``onehot`` 模式在线编码为 ``(5, seq_len)`` tensor；
- ``kmer`` 模式转为整数 token 序列；
- ``llm`` 模式直接读取预计算向量；
- ``DummyEPIDataset`` 生成形状真实的随机张量，方便全链路调试。

为什么在线编码重要
+++++++++++++++++++++

相较于旧版，当前工程上的关键升级之一是移除 PNG 中间表示。在当前设计中：

- 序列字符串是唯一的规范存储形式；
- one-hot tensor 只在 batch 加载时临时生成；
- 避免了量化损失、图片 I/O 开销和巨量临时文件。

验证逻辑
+++++++++++++++

训练期间每个 epoch 都会配套验证：

- 损失函数使用 ``BCEWithLogitsLoss``；
- 通过 ``sigmoid`` 得到概率；
- 用 AUROC 作为模型选择主指标；
- 当 AUROC 长时间不提升时触发 early stopping。

评估输出
+++++++++++++++

``python -m src.evaluate`` 会输出：

- AUROC；
- AUPRC；
- F1；
- Accuracy；
- ROC 曲线图；
- PR 曲线图；
- 每个 seed 的 ``metrics.json``；
- 多个 seed 汇总后的 ``summary.json``。

指标解释
+++++++++++++++

这些指标更符合调控基因组任务的实际需求：

- AUROC 衡量整体排序能力；
- AUPRC 在类别不均衡时更有代表性；
- F1 反映 precision 与 recall 的平衡；
- Accuracy 保留为直观基线指标。

推荐流程
+++++++++++++++

1. 先对一个细胞系做 ``train/val/test`` 预处理；
2. 如果环境刚搭好，先运行 ``scripts/test_pipeline.py``；
3. 先跑单个 seed 做 sanity check；
4. 再启动五个 seed 的正式实验；
5. 评估并收集每个 seed 的结果；
6. 对比模型家族和融合策略。

.. image:: ../img/div.png
