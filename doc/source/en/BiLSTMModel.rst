BiLSTM Model
============

Switch language: :doc:`../zh/BiLSTMModel`

``M4`` is the bidirectional LSTM encoder of the project.

Why BiLSTM is included
++++++++++++++++++++++

Bidirectional recurrent models are a natural counterpart to CNNs:

- CNNs focus on local pattern extraction;
- BiLSTMs explicitly process the sequence as an ordered chain;
- forward and backward hidden states capture left and right context jointly.

Architecture summary
++++++++++++++++++++

- Input: one-hot tensor ``(B, 5, L)``
- Sequence layout: transposed into ``(B, L, 5)``
- Core: 2-layer bidirectional LSTM
- Hidden width: 256 per direction
- Output: concatenated forward/backward final states

Why this helps for EPI
++++++++++++++++++++++

Enhancer and promoter regions are long sequences where regulatory information is
not confined to a single motif window. A BiLSTM can model:

- ordered motif progression;
- local-to-mid-range contextual accumulation;
- asymmetric sequence signals that plain pooling may wash out.

Strengths
+++++++++

- strong sequence-order awareness;
- intuitive recurrent inductive bias;
- useful comparator against attention-based and linear-time models.

Limitations
+++++++++++

- recurrent processing is inherently more sequential than convolution or fully
  parallel attention alternatives;
- very long genomic sequences can become computationally expensive.

Project role
++++++++++++

``M4`` is important because it represents the classic recurrent baseline in the
benchmark. It helps answer whether explicit sequential recurrence remains useful
after introducing Transformer-style, state-space, and foundation-model routes.

.. image:: ../img/div.png

