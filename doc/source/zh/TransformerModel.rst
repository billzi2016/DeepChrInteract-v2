Transformer 模型
================

切换语言: :doc:`../en/TransformerModel`

``M6`` 是项目中的标准 Transformer 编码路线。

为什么标准 Transformer 仍然必须存在
+++++++++++++++++++++++++++++++++++

即便已经有很多更新的变体，标准 Transformer 仍然是关键参考，因为它提供：

- 显式的全局 token-to-token 交互；
- 最容易理解的注意力基线；
- 用于比较 Linear Transformer、iTransformer、MAE 的中心参照点。

结构概览
+++++++++++++++++++++++++++++

- 输入：one-hot 序列
- 前端：先用 CNN 做降采样
- token 形式：压缩后的序列加 CLS token
- 编码器：堆叠 TransformerEncoderLayer
- 输出：CLS 表示

它的核心注意力算子可以写成：

.. math::

   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

这意味着每个 token 都会根据内容，自适应地汇总其他 token 的信息，因此 Transformer
才会成为最标准的全局交互模型。

为什么 CNN 前端重要
+++++++++++++++++++

如果直接对 10kbp 单碱基序列做二次复杂度注意力，代价会很高。CNN 前端先压缩 token
数，可以让模型保留全局注意力优势，同时避免在原始长度上直接支付完整二次代价。

从功能上看，CNN 前端扮演的是局部压缩器：先把 motif 级局部信号提炼成更短的 token
序列，再让注意力层在这些压缩单元之间建模更大范围依赖。

优势
+++

- 全局交互表达能力强；
- 是最标准的注意力比较基线；
- 也为 MAE 预训练路线提供了天然基础。

局限
+++

- 相比线性时间路线仍然更重；
- 依赖降采样，可能压缩掉部分细粒度信息。

.. image:: ../img/div.png
