DeepChrInteract Model Overview
==============================

Switch language: :doc:`../zh/DeepChrInteract`

This page serves as the navigation hub for the model system rather than trying
to explain every architecture in a single long document. The project spans
classical convolutional baselines, recurrent sequence models, attention-based
architectures, state-space style encoders, DNA foundation models, and
self-supervised pretraining.

Model family map
++++++++++++++++

- Classical convolutional encoders:
  :doc:`CNNModels`
- Residual convolutional branch:
  :doc:`ResNetModels`
- Recurrent encoders:
  :doc:`BiLSTMModel`, :doc:`mLSTMModel`
- Attention family:
  :doc:`TransformerModel`, :doc:`LinearTransformerModel`,
  :doc:`iTransformerModel`
- Linear-time and state-space style models:
  :doc:`MambaModel`, :doc:`RWKVModel`
- Hybrid architectures:
  :doc:`HybridModels`
- Foundation-model and self-supervised routes:
  :doc:`DNAFoundationModels`, :doc:`MAEModel`
- Pairwise representation layer:
  :doc:`FusionStrategies`

Registry snapshot
+++++++++++++++++

- ``M1``: CNN single-branch
  Simplest local motif baseline. Detail page: :doc:`CNNModels`
- ``M2``: CNN dual-branch
  Separate enhancer/promoter encoding. Detail page: :doc:`CNNModels`
- ``M3``: k-mer + CNN
  Token embedding plus convolution. Detail page: :doc:`CNNModels`
- ``ResNet18 / ResNet34``
  Deeper residual convolution route. Detail page: :doc:`ResNetModels`
- ``M4``: BiLSTM
  Bidirectional sequential dependency modeling. Detail page: :doc:`BiLSTMModel`
- ``M5``: mLSTM
  Matrix-memory recurrent modeling. Detail page: :doc:`mLSTMModel`
- ``M6``: Transformer
  Global attention after CNN compression. Detail page: :doc:`TransformerModel`
- ``M7``: Linear Transformer
  Kernelized linear attention. Detail page: :doc:`LinearTransformerModel`
- ``M8``: iTransformer
  Channel-wise attention reinterpretation. Detail page: :doc:`iTransformerModel`
- ``M9``: Mamba
  Selective state-space sequence modeling. Detail page: :doc:`MambaModel`
- ``M10``: RWKV
  Linear-recurrence time mixing. Detail page: :doc:`RWKVModel`
- ``M11``: CNN + BiLSTM
  Local detector plus recurrent context. Detail page: :doc:`HybridModels`
- ``M12``: CNN + Transformer
  Local detector plus global attention. Detail page: :doc:`HybridModels`
- ``M13``: DNA LLM encoder
  External pretrained genomic representations. Detail page: :doc:`DNAFoundationModels`
- ``M14``: MAE-pretrained Transformer
  Self-supervised pretraining on project data. Detail page: :doc:`MAEModel`

Design logic
++++++++++++

The project is intentionally broad. It is meant to expose a continuum of
sequence modeling assumptions:

- local pattern extraction through CNNs;
- deeper residual convolution as the natural extension of the CNN family;
- ordered dependency modeling through recurrent cells;
- global interaction modeling through attention;
- long-context efficiency through linear-time or state-space style models;
- transfer learning through pretrained genomic foundation models;
- task-adapted representation learning through MAE-style pretraining.

This organization makes it easier to compare not only raw performance, but also
inductive bias, runtime behavior, memory tradeoffs, and data efficiency.

.. image:: ../img/div.png
