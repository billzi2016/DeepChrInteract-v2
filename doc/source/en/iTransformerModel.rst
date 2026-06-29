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

Strengths
+++++++++

- very efficient on one-hot inputs;
- conceptually different from both CNN and standard attention routes;
- useful as an architectural reinterpretation rather than a simple scaling tweak.

Limitations
+++++++++++

- it trades away the classic token-to-token attention view;
- effectiveness depends on whether cross-channel structure is a sufficiently
  strong signal carrier for the task.

.. image:: ../img/div.png

