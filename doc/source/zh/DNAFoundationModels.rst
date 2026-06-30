DNA 基础模型
============

切换语言: :doc:`../en/DNAFoundationModels`

``M13`` 是项目中的 DNA foundation model 路线。

支持的 backbone
+++++++++++++++

- DNABERT
- DNABERT-2
- Nucleotide Transformer
- HyenaDNA

为什么这一类模型重要
++++++++++++++++++++

这类模型把外部大规模预训练引入 EPI 任务。也就是说，它不只是依靠当前项目数据
从零开始学，而是把更广泛基因组语料中学到的先验表示带进来。

两种使用模式
+++++++++++++++++++++++++++++

- frozen：冻结 backbone，只训练投影层和任务头
- finetune：以较小学习率继续更新 backbone

如果用表示学习流水线来写，它其实就是：

.. math::

   \text{DNA sequence} \rightarrow \text{tokenizer} \rightarrow \text{pretrained backbone}
   \rightarrow \text{sequence embedding} \rightarrow \text{task head}

它要回答的核心问题是：大规模基因组预训练是否已经学到了足够强的调控先验，从而让项目级
数据集不必再从零开始学习全部表示。

为什么要保留多个 backbone
+++++++++++++++++++++++++

这些模型在分词方式、上下文长度、预训练数据和扩展路线方面都不同：

- DNABERT 强调 k-mer token 化和早期基因组语言建模；
- DNABERT-2 在 tokenizer 和训练设计上更现代；
- Nucleotide Transformer 强调大规模基因组预训练；
- HyenaDNA 强调长上下文和线性时间风格。

项目中的作用
+++++++++++++++++++++++++++++

这一页让整个 benchmark 可以系统比较：

- 手工编码和从零训练；
- 任务内训练和迁移学习；
- 外部预训练先验对 EPI 任务的增益。

计算复杂度与适用范围
+++++++++++++++++++++

- 时间复杂度：高度依赖 backbone，本身可能从中等规模 k-mer 语言模型一路延伸到更重的长上下文预训练编码器。
- 显存特征：通常是本仓库里最重的一类，尤其在 finetune 模式下，需要保留完整预训练骨干的可训练状态。
- 适用范围：适合把外部基因组先验作为重点、并且硬件预算允许更大模型运行的场景。

.. image:: ../img/div.png
