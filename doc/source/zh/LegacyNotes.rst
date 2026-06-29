旧版保留说明
============

切换语言: :doc:`../en/LegacyNotes`

本页用于保留旧版文档中的核心写作内容，而不是将其抹掉。

旧版 QuickStart 重点
+++++++++++++++++++++++

- CPU 内存建议 ``16GB``
- GPU 显存建议 ``8GB``
- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0

旧版动机
+++++++++++++++

旧版文档的核心动机是：虽然基于染色质相互作用区域侧翼 DNA 序列的深度学习方法
不断发展，但能够整合并评估不同深度学习架构的综合工具仍然不足。

旧版模型清单
+++++++++++++++

- ``onehot_cnn_one_branch``
- ``onehot_cnn_two_branch``
- ``onehot_embedding_dense``
- ``onehot_embedding_cnn_one_branch``
- ``onehot_embedding_cnn_two_branch``
- ``onehot_dense``
- ``onehot_resnet18``
- ``embedding_cnn_one_branch``
- ``embedding_cnn_two_branch``

旧版数据组织重点
+++++++++++++++++++++++++++++++++++++++

原始项目主要围绕以下对象：

- ``DeepChrInteract.py`` 主入口；
- ``data_preprocessing.py`` 预处理；
- ``model.py`` 模型定义；
- ``train.py`` 训练；
- ``test.py`` 测试；
- ``embedding_matrix.npy`` DNA2Vec 特征；
- ``data/`` 数据目录；
- ``h5_weights/`` 和 ``result/`` 输出目录。

保留原则
+++++++++++++++

当前文档会保留这些旧版精华，并在其上扩展新的 PyTorch 设计，而不是覆盖掉原来的
方法脉络。

.. image:: ../img/div.png
