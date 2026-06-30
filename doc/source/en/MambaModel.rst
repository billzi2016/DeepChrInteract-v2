Mamba Model
===========

Switch language: :doc:`../zh/MambaModel`

``M9`` is the project's Mamba-based selective state-space encoder.

Why Mamba is included
+++++++++++++++++++++

Mamba is one of the most important modern alternatives to attention for long
sequences. It keeps a strong sequence-modeling flavor while aiming for linear
scaling and input-dependent state transitions.

Project implementation
++++++++++++++++++++++

- Input: one-hot sequence projected into model space
- Preferred runtime: official ``mamba-ssm`` package
- Fallback runtime: PyTorch approximation that preserves pipeline usability
- Output: sequence mean pooling after stacked blocks

At a high level, the selective state-space update can be summarized as:

.. math::

   h_t = \bar{A}(x_t) h_{t-1} + \bar{B}(x_t) x_t, \qquad
   y_t = C(x_t) h_t

Unlike fixed-coefficient state-space models, the transition depends on the
current input, which is why Mamba can adapt its memory behavior to sequence
content.

Why this matters in practice
++++++++++++++++++++++++++++

The project uses Mamba not only because it is popular, but because it tests a
specific hypothesis: long-range genomic dependencies may benefit from
state-space style modeling without the memory profile of full attention.

Strengths
+++++++++

- linear-time flavor on long sequences;
- modern alternative to Transformer scaling;
- relevant for comparing sequence efficiency against RWKV and Linear
  Transformer.

Computational complexity
++++++++++++++++++++++++

- Time: designed for linear-time sequence processing, making it suitable for
  much longer contexts than dense quadratic attention.
- Memory: favorable for long inputs because it avoids explicit token-token
  attention maps and instead carries state forward recurrently.
- Best-fit regime: strong candidate for long genomic windows where preserving
  sequence context matters but attention memory would be excessive.

Caveat
++++++

The highest-fidelity behavior depends on the official ``mamba-ssm`` runtime.
The fallback path is useful for portability and pipeline validation, but should
not be treated as identical to the optimized implementation.

.. image:: ../img/div.png
