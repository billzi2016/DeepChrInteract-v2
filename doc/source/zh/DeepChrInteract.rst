DeepChrInteract 模型与方法
==========================

切换语言: :doc:`../en/DeepChrInteract`

集成深度学习模型
+++++++++++++++++++++++++++++++++++++++

旧版文档中已经包含多种模型，例如：

- ``onehot_cnn_one_branch``
- ``onehot_cnn_two_branch``
- ``onehot_embedding_dense``
- ``onehot_embedding_cnn_one_branch``
- ``onehot_embedding_cnn_two_branch``
- ``onehot_dense``
- ``onehot_resnet18``
- ``embedding_cnn_one_branch``
- ``embedding_cnn_two_branch``

当前项目保留这种“多模型可比较”的核心价值，并扩展为统一的 PyTorch 模型注册表。

模型注册表
++++++++++

A 组：经典 CNN 基线

- ``M1``：单路 CNN 基线
- ``M2``：双路 CNN 基线
- ``M3``：k-mer embedding + CNN

B 组：循环序列模型

- ``M4``：双向 LSTM
- ``M5``：mLSTM / xLSTM 风格模型

C 组：注意力模型

- ``M6``：带 CNN 前端的标准 Transformer
- ``M7``：Linear Transformer
- ``M8``：iTransformer

D 组：线性递推 / 状态空间类模型

- ``M9``：Mamba
- ``M10``：RWKV

E 组：混合与预训练编码器

- ``M11``：CNN + BiLSTM
- ``M12``：CNN + Transformer
- ``M13``：DNA LLM encoder
- ``M14``：MAE 预训练 Transformer

融合策略
++++++++

框架支持六种 enhancer-promoter 融合方式：

- ``concat``
- ``add``
- ``subtract``
- ``multiply``
- ``bilinear``
- ``concat_sub_mul`` (默认)

当前实现原则
++++++++++++

- 所有模型通过 ``build_model(model_id, config)`` 统一构建；
- 训练接口统一为 enhancer / promoter 配对输入；
- 每次实验保存 JSON 配置快照；
- 评估指标使用 AUROC、AUPRC、F1、Accuracy；
- 不再采用旧版 PNG 渲染流程。

训练流程特征
++++++++++++

当前训练循环包含：

- Adam 优化器；
- cosine warm restarts 调度；
- 梯度裁剪；
- 基于验证集 AUROC 的 early stopping；
- ``best.pt`` 与 ``last.pt`` checkpoint；
- 支持 dummy 模式做无数据完整性验证。

现代化改造的意义
++++++++++++++++

这不只是代码迁移，而是方法层面的扩展：

- 保留经典卷积模型，方便与旧版结果或思路对齐；
- 增加 RNN、Transformer、线性时间模型与状态空间模型；
- 引入 DNA foundation model embedding，对比手工编码与预训练表示；
- 通过 MAE 探索标签较少场景下的序列表征学习。

.. image:: ../img/div.png
