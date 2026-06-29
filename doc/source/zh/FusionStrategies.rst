融合策略
========

切换语言: :doc:`../en/FusionStrategies`

增强子-启动子预测不只是单序列编码问题，还涉及两条序列编码结果如何在分类前组合。

支持的策略
+++++++++++++++++++++++++++++

- ``concat``
- ``add``
- ``subtract``
- ``multiply``
- ``bilinear``
- ``concat_sub_mul``

如果把 enhancer / promoter 表示写成 :math:`h_e` 和 :math:`h_p`，那么主要融合形式
可以写成：

.. math::

   \mathrm{concat}: [h_e; h_p]

.. math::

   \mathrm{add}: h_e + h_p

.. math::

   \mathrm{subtract}: h_e - h_p

.. math::

   \mathrm{multiply}: h_e \odot h_p

.. math::

   \mathrm{concat\_sub\_mul}: [h_e; h_p; h_e - h_p; h_e \odot h_p]

为什么融合层重要
+++++++++++++++++++++++++++++

即便单路编码器很强，如果 pairwise interaction 层太弱，最终性能仍可能受限。
融合层决定了 enhancer 与 promoter 之间的关系信息，能以多丰富的形式暴露给分类器。

所以融合层不应该被看成无关紧要的小实现细节。它直接决定了分类器到底能看到哪些关系结构。

主要策略的含义
+++++++++++++++++++++++++++++

- ``concat``：保留两条向量原貌，把关系学习交给分类头
- ``add``：强调共享强度与对称聚合
- ``subtract``：显式暴露方向差异
- ``multiply``：强调逐元素协同与一致性
- ``bilinear``：用可学习双线性交互表示更复杂关系
- ``concat_sub_mul``：把身份、差异、一致性三种信息同时保留

默认选择
+++++++++++++++++++++++++++++

当前项目默认使用 ``concat_sub_mul``，因为它在表示丰富度和参数代价之间取得了比较平衡的折中。

.. image:: ../img/div.png
