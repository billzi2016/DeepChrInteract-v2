RWKV Model
==========

Switch language: :doc:`../zh/RWKVModel`

``M10`` is the RWKV-style linear-recurrence encoder in the project.

What RWKV contributes
+++++++++++++++++++++

RWKV is interesting because it mixes two design worlds:

- recurrent time evolution;
- Transformer-like channel mixing.

In effect, it offers a way to model long sequences with recurrence-inspired
updates while retaining feed-forward style expressive capacity.

Project implementation highlights
+++++++++++++++++++++++++++++++++

- Input: projected one-hot sequence
- Temporal component: time-mixing with learned decay
- Channel component: per-position feed-forward style mixing
- Stabilization: log-space handling in the recurrent weighting path

Why this is relevant for EPI
++++++++++++++++++++++++++++

Enhancer-promoter prediction needs more than isolated motif hits. RWKV offers an
alternative way to accumulate sequence evidence across long contexts while
avoiding classic quadratic attention.

Strengths
+++++++++

- long-context-friendly design;
- distinct comparison point relative to Mamba;
- hybrid recurrent/feed-forward character.

Limitations
+++++++++++

- less conventional than BiLSTM or standard Transformer;
- requires careful numerical treatment for stable sequence accumulation.

.. image:: ../img/div.png

