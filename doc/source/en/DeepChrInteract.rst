DeepChrInteract Models and Methods
==================================

Switch language: :doc:`../zh/DeepChrInteract`

Integrated Deep learning models
+++++++++++++++++++++++++++++++

The legacy release included multiple models such as
``onehot_cnn_one_branch``, ``onehot_cnn_two_branch``,
``onehot_embedding_dense``, ``onehot_embedding_cnn_one_branch``,
``onehot_embedding_cnn_two_branch``, ``onehot_dense``,
``onehot_resnet18``, ``embedding_cnn_one_branch``, and
``embedding_cnn_two_branch``.

The current project preserves that comparative spirit and expands the model
family to fourteen encoder variants under a unified PyTorch registry.

Model registry
++++++++++++++

Group A: classical CNN baselines

- ``M1``: CNN single-branch baseline
- ``M2``: CNN dual-branch baseline
- ``M3``: k-mer embedding plus CNN

Group B: recurrent sequence models

- ``M4``: bidirectional LSTM
- ``M5``: mLSTM / xLSTM-style sequence model

Group C: attention-based models

- ``M6``: standard Transformer with CNN frontend
- ``M7``: Linear Transformer
- ``M8``: iTransformer

Group D: linear-recurrence and state-space style models

- ``M9``: Mamba
- ``M10``: RWKV

Group E: hybrid and pretrained encoders

- ``M11``: CNN + BiLSTM
- ``M12``: CNN + Transformer
- ``M13``: DNA LLM encoder
- ``M14``: MAE-pretrained Transformer

Fusion strategies
+++++++++++++++++

The framework supports six enhancer-promoter fusion strategies:

- ``concat``
- ``add``
- ``subtract``
- ``multiply``
- ``bilinear``
- ``concat_sub_mul`` (default)

Current implementation principles
+++++++++++++++++++++++++++++++++

- all models are constructed through ``build_model(model_id, config)``;
- the training interface is unified around paired enhancer/promoter input;
- experiment configuration is stored as JSON for reproducibility;
- evaluation uses AUROC, AUPRC, F1, and accuracy rather than accuracy alone;
- the PNG rendering path used in the legacy system is removed.

Training behavior
+++++++++++++++++

The training loop includes:

- Adam optimization;
- cosine warm restarts scheduling;
- gradient clipping;
- validation AUROC-driven early stopping;
- ``best.pt`` and ``last.pt`` checkpointing;
- optional dummy mode for end-to-end pipeline verification.

Why the modernization matters
+++++++++++++++++++++++++++++

The project is not only a code migration. It is a methodological expansion:

- classical local convolutional inductive bias is retained for baseline
  reproducibility;
- recurrent, attention-based, and linear-time architectures are added for
  longer-range sequence dependency modeling;
- DNA foundation model embeddings allow comparison between handcrafted encoding
  schemes and pretrained genomic representations;
- MAE-style pretraining explores label-efficient representation learning for
  regulatory genomics.

.. image:: ../img/div.png

