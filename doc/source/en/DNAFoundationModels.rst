DNA Foundation Models
=====================

Switch language: :doc:`../zh/DNAFoundationModels`

``M13`` is the DNA foundation-model branch of the project.

Supported backbones
+++++++++++++++++++

- DNABERT
- DNABERT-2
- Nucleotide Transformer
- HyenaDNA

Why this family matters
+++++++++++++++++++++++

These models bring external large-scale pretraining into the EPI task. Instead
of learning from scratch on the project dataset alone, they inject prior genomic
sequence knowledge learned from broader corpora.

Two usage modes
+++++++++++++++

- frozen mode: use pretrained features and only train the projection and task
  head;
- finetune mode: update the backbone with a smaller learning rate.

Why multiple backbones are useful
+++++++++++++++++++++++++++++++++

The supported models differ in tokenization, context assumptions, pretraining
data, and scaling behavior:

- DNABERT emphasizes k-mer tokenization and early genomic language modeling;
- DNABERT-2 modernizes tokenizer and training design;
- Nucleotide Transformer emphasizes large-scale genomic pretraining;
- HyenaDNA focuses on long-context sequence modeling with linear-time flavor.

Project role
++++++++++++

This family lets the benchmark compare handcrafted encoders against transfer
learning. It asks whether external genomic prior knowledge can outperform or
complement task-specific training from raw sequence encodings.

.. image:: ../img/div.png

