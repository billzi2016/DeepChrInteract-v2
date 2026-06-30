iTransformer Model
==================

Switch language: :doc:`../zh/iTransformerModel`

``M8`` adapts the iTransformer idea to genomic one-hot sequence inputs.

Why it is unusual
+++++++++++++++++

The original iTransformer idea came from time-series modeling and inverts the
usual attention perspective. Instead of attending primarily across positions, it
attends across variables. In this repository, the five nucleotide channels
(``A/C/G/T/N``) are treated as the compact variable axis.

Why this is interesting for DNA
+++++++++++++++++++++++++++++++

One-hot genomic input has only five channels, which makes channel-wise
attention extremely cheap compared with position-wise quadratic attention. This
creates a distinctive model:

- channel interactions are modeled explicitly;
- position-wise nonlinear transformation is still retained;
- complexity is driven by channel count rather than raw sequence length.

The conceptual inversion is the main point. Instead of treating positions as
tokens and channels as features inside each token, iTransformer lets the
channels become the attended entities while sequence positions behave more like
the observation axis over which those variables are organized.

Strengths
+++++++++

- very efficient on one-hot inputs;
- conceptually different from both CNN and standard attention routes;
- useful as an architectural reinterpretation rather than a simple scaling tweak.

Computational complexity
++++++++++++++++++++++++

- Time: attention cost is driven more by channel dimension than by raw sequence
  length, which is unusually favorable for five-channel one-hot genomic input.
- Memory: typically light relative to full position-wise attention because the
  expensive all-position interaction matrix is avoided.
- Best-fit regime: especially attractive when the representation remains close
  to raw one-hot channels and a channel-centric inductive bias is desirable.

Limitations
+++++++++++

- it trades away the classic token-to-token attention view;
- effectiveness depends on whether cross-channel structure is a sufficiently
  strong signal carrier for the task.

.. image:: ../img/div.png
