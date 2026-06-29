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

Why the CNN frontend is important
+++++++++++++++++++++++++++++++++

Direct quadratic attention over full 10kbp single-base sequences is expensive.
The CNN frontend reduces token count before the Transformer stage, allowing the
model to keep the benefits of global attention without paying the full
quadratic cost on the raw sequence length.

Strengths
+++++++++

- strong global interaction modeling;
- highly interpretable as the canonical attention baseline;
- natural foundation for MAE-style pretraining reuse.

Limitations
+++++++++++

- still more expensive than linear-time alternatives;
- relies on downsampling, which may compress away some fine-grained detail.

.. image:: ../img/div.png

