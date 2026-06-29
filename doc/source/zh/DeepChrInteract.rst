DeepChrInteract 模型总览
========================

切换语言: :doc:`../en/DeepChrInteract`

本页作为模型体系的导航页，而不是把全部架构塞进一个超长页面。当前项目覆盖了
经典卷积基线、循环序列模型、注意力模型、状态空间风格编码器、DNA 基础模型、
以及自监督预训练路线。

模型地图
+++++++++++++++++++++++++++++

- 经典卷积编码器：:doc:`CNNModels`
- 残差卷积路线：:doc:`ResNetModels`
- 循环模型：:doc:`BiLSTMModel`、:doc:`mLSTMModel`
- 注意力家族：:doc:`TransformerModel`、:doc:`LinearTransformerModel`、
  :doc:`iTransformerModel`
- 线性时间 / 状态空间风格模型：:doc:`MambaModel`、:doc:`RWKVModel`
- 混合架构：:doc:`HybridModels`
- 基础模型与自监督路线：:doc:`DNAFoundationModels`、:doc:`MAEModel`
- enhancer-promoter 配对层：:doc:`FusionStrategies`

注册表概览
+++++++++++++++++++++++++++++

- ``M1``：单路 CNN
  最基础的局部 motif baseline。详情页：:doc:`CNNModels`
- ``M2``：双路 CNN
  enhancer / promoter 分开编码。详情页：:doc:`CNNModels`
- ``M3``：k-mer + CNN
  token embedding 加卷积。详情页：:doc:`CNNModels`
- ``ResNet18 / ResNet34``
  更深的残差卷积路线。详情页：:doc:`ResNetModels`
- ``M4``：BiLSTM
  双向顺序依赖建模。详情页：:doc:`BiLSTMModel`
- ``M5``：mLSTM
  矩阵记忆递归建模。详情页：:doc:`mLSTMModel`
- ``M6``：Transformer
  CNN 压缩后做全局注意力。详情页：:doc:`TransformerModel`
- ``M7``：Linear Transformer
  核函数化线性注意力。详情页：:doc:`LinearTransformerModel`
- ``M8``：iTransformer
  通道维重解释注意力。详情页：:doc:`iTransformerModel`
- ``M9``：Mamba
  选择性状态空间建模。详情页：:doc:`MambaModel`
- ``M10``：RWKV
  线性递推 time mixing。详情页：:doc:`RWKVModel`
- ``M11``：CNN + BiLSTM
  局部检测器加循环上下文。详情页：:doc:`HybridModels`
- ``M12``：CNN + Transformer
  局部检测器加全局注意力。详情页：:doc:`HybridModels`
- ``M13``：DNA LLM encoder
  外部预训练基因组表示。详情页：:doc:`DNAFoundationModels`
- ``M14``：MAE 预训练 Transformer
  在项目数据上的自监督预训练。详情页：:doc:`MAEModel`

设计逻辑
+++++++++++++++++++++++++++++

这个项目的广度是有意为之，用来展示一条完整的序列建模光谱：

- CNN 负责局部模式与 motif 抽取；
- 更深的残差卷积负责把卷积路线继续推深；
- 循环模型负责顺序依赖；
- 注意力模型负责全局交互；
- 线性时间 / 状态空间路线负责长序列效率；
- 基础模型负责迁移学习；
- MAE 路线负责贴近任务分布的自监督表征学习。

这种结构不仅便于比较指标，也便于比较归纳偏置、运行复杂度、显存开销和数据效率。

.. image:: ../img/div.png
