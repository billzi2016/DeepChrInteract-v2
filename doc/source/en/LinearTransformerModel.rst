Linear Transformer Model
========================

Switch language: :doc:`../zh/LinearTransformerModel`

``M7`` is the linear-attention alternative to the standard Transformer.

Key motivation
++++++++++++++

Standard attention has quadratic dependence on sequence length. For genomic
sequences, that can become a serious bottleneck. The linear Transformer replaces
the softmax attention form with a kernelized approximation so that attention can
be computed in linear-time style form with respect to sequence length.

Project implementation highlights
+++++++++++++++++++++++++++++++++

- Input: one-hot sequence tensor projected into model space
- Positional encoding: sinusoidal
- Attention: ELU+1 kernel feature map
- Pooling: mean pooling over sequence positions

The key approximation replaces softmax attention with feature maps:

.. math::

   \mathrm{Attn}(Q, K, V) \approx
   \frac{\phi(Q)\big(\phi(K)^\top V\big)}
        {\phi(Q)\big(\phi(K)^\top \mathbf{1}\big)}

where :math:`\phi(\cdot)` is a positive kernel feature map such as ELU+1. This
reorders the computation so sequence length scaling becomes linear in style.

Why it matters for EPI
++++++++++++++++++++++

This model asks an important question: can we keep the global interaction flavor
of attention while scaling better to long DNA sequences than a standard
Transformer?

That question matters directly for genomic inputs because long-range regulatory
dependencies are scientifically relevant, but full quadratic attention becomes
costly exactly in the length regime where those dependencies matter most.

Strengths
+++++++++

- better scaling behavior than quadratic attention;
- no mandatory CNN token compression step;
- useful for long-range sequence modeling studies.

Limitations
+++++++++++

- linearized attention is an approximation, not a drop-in perfect substitute
  for full softmax attention;
- quality depends on whether the approximation preserves the interactions most
  relevant to the task.

.. image:: ../img/div.png
