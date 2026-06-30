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

At each step, an LSTM updates gates and memory through:

.. math::

   i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)

.. math::

   f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)

.. math::

   o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)

.. math::

   \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)

.. math::

   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad
   h_t = o_t \odot \tanh(c_t)

The bidirectional encoder runs this recurrence in both directions so that each
output representation can reflect both upstream and downstream context.

Why this helps for EPI
++++++++++++++++++++++

Enhancer and promoter regions are long sequences where regulatory information is
not confined to a single motif window. A BiLSTM can model:

- ordered motif progression;
- local-to-mid-range contextual accumulation;
- asymmetric sequence signals that plain pooling may wash out.

This matters when the functional meaning of a motif depends on order, nearby
context, or the sequence of several local events rather than on isolated hits.

Strengths
+++++++++

- strong sequence-order awareness;
- intuitive recurrent inductive bias;
- useful comparator against attention-based and linear-time models.

Computational complexity
++++++++++++++++++++++++

- Time: sequential recurrence gives roughly :math:`O(L \cdot H^2)` behavior per
  layer, with limited parallelism across positions.
- Memory: moderate for hidden states, but training cost rises with sequence
  length because backpropagation must preserve recurrent activations across the
  chain.
- Best-fit regime: useful for short-to-medium windows where ordered context is
  important and full global attention would be unnecessary or overly expensive.

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
