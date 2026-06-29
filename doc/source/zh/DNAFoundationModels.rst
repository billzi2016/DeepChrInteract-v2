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

.. image:: ../img/div.png
