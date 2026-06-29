Hybrid Models
=============

Switch language: :doc:`../zh/HybridModels`

This page covers the two mixed-architecture models: ``M11`` and ``M12``.

Why hybrids exist
+++++++++++++++++

Hybrid models are designed around a practical intuition: different architectural
families are good at different things.

- CNNs are good at local motif extraction.
- BiLSTMs are good at sequential context accumulation.
- Transformers are good at global interaction modeling.

Hybrid models try to combine those strengths rather than forcing a single family
to do everything.

M11: CNN + BiLSTM
+++++++++++++++++

``M11`` uses a CNN frontend followed by a bidirectional LSTM.

- CNN role: compress the raw sequence and detect local motif-like patterns
- BiLSTM role: model dependencies over the compressed feature sequence

This is attractive when the raw sequence is too long for direct recurrent
modeling but ordered context still matters.

M12: CNN + Transformer
++++++++++++++++++++++

``M12`` uses the same CNN-style idea as a frontend, but hands the compressed
sequence to a Transformer encoder.

- CNN role: reduce length and expose local features
- Transformer role: perform global interaction modeling on a shorter sequence

This is a pragmatic compromise between raw-sequence Transformers and purely
convolutional baselines.

Why hybrids matter for the benchmark
++++++++++++++++++++++++++++++++++++

They test whether architectural composition is better than architectural purity
for this task. In many practical sequence problems, the answer is often yes:

- local detectors handle motif discovery efficiently;
- later modules reason over the more abstract feature stream.

.. image:: ../img/div.png

