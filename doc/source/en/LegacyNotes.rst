Legacy Notes
============

Switch language: :doc:`../zh/LegacyNotes`

This page preserves the essential writing from the original documentation rather
than replacing it.

Original QuickStart highlights
++++++++++++++++++++++++++++++

- CPU memory is recommended as ``16GB``
- GPU memory is recommended as ``8GB``
- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0

Original motivation
+++++++++++++++++++

Though deep learning methods have been widely developed for predicting chromatin
interactions using flanking DNA sequence in identified chromatin interaction
regions, a comprehensive software toolkit to integrate and evaluate different
deep learning architectures are under-developed.

Original model inventory
++++++++++++++++++++++++

- ``onehot_cnn_one_branch``
- ``onehot_cnn_two_branch``
- ``onehot_embedding_dense``
- ``onehot_embedding_cnn_one_branch``
- ``onehot_embedding_cnn_two_branch``
- ``onehot_dense``
- ``onehot_resnet18``
- ``embedding_cnn_one_branch``
- ``embedding_cnn_two_branch``

Original data organization highlights
+++++++++++++++++++++++++++++++++++++

The original project centered on:

- ``DeepChrInteract.py`` as the main entry file;
- ``data_preprocessing.py`` for preprocessing;
- ``model.py`` for model definitions;
- ``train.py`` for training;
- ``test.py`` for evaluation;
- ``embedding_matrix.npy`` for pretrained DNA2Vec features;
- ``data/`` for labeled sequence folders;
- ``h5_weights/`` and ``result/`` for saved outputs.

Preservation policy
+++++++++++++++++++

The current documentation keeps these legacy concepts visible and extends them
with the modern PyTorch redesign rather than erasing them.

.. image:: ../img/div.png

