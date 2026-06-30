Transformer Model
=================

Switch language: :doc:`../zh/TransformerModel`

``M6`` is the standard Transformer encoder route in the project.

Why a standard Transformer is still necessary
+++++++++++++++++++++++++++++++++++++++++++++

Even with many newer variants available, a standard Transformer remains a key
reference model because it provides:

- explicit global token-to-token interaction;
- a widely understood attention baseline;
- a central comparison point for linear attention, iTransformer, and MAE routes.

Architecture summary
++++++++++++++++++++

- Input: one-hot sequence tensor
- Frontend: CNN downsampling before attention
- Sequence tokenization: compressed token stream plus a learnable CLS token
- Encoder: stacked Transformer encoder layers
- Output: CLS representation

The core attention operator is:

.. math::

   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

This gives each token a content-dependent weighted combination of all other
tokens, which is why the Transformer is the canonical global-interaction model.

Why the CNN frontend is important
+++++++++++++++++++++++++++++++++

Direct quadratic attention over full 10kbp single-base sequences is expensive.
The CNN frontend reduces token count before the Transformer stage, allowing the
model to keep the benefits of global attention without paying the full
quadratic cost on the raw sequence length.

Conceptually, the CNN frontend acts as a local compression layer: motif-scale
signals are distilled into a shorter token sequence before the attention module
models wider-range dependencies among those compressed units.

Strengths
+++++++++

- strong global interaction modeling;
- highly interpretable as the canonical attention baseline;
- natural foundation for MAE-style pretraining reuse.

Computational complexity
++++++++++++++++++++++++

- Time: standard self-attention is quadratic in token count,
  :math:`O(T^2 \cdot d)`, which is why the CNN frontend is used to shorten the
  sequence before attention.
- Memory: attention maps also scale quadratically with token count, making this
  the main bottleneck for long genomic inputs.
- Best-fit regime: appropriate when the compressed token length is moderate and
  explicit global interaction is worth the cost.

Limitations
+++++++++++

- still more expensive than linear-time alternatives;
- relies on downsampling, which may compress away some fine-grained detail.

.. image:: ../img/div.png
