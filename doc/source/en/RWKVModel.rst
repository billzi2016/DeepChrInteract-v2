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

Its weighted key-value accumulation can be viewed schematically as a decayed
running summary:

.. math::

   s_t = \alpha_t \odot s_{t-1} + \beta_t \odot v_t, \qquad
   y_t = \frac{s_t}{z_t}

where the decay terms control how much old evidence is retained and the
normalization path keeps the recurrent aggregation numerically stable.

Why this is relevant for EPI
++++++++++++++++++++++++++++

Enhancer-promoter prediction needs more than isolated motif hits. RWKV offers an
alternative way to accumulate sequence evidence across long contexts while
avoiding classic quadratic attention.

This makes RWKV attractive when the task depends on gradual evidence
accumulation across many positions rather than only on a few sharp token-token
links.

Strengths
+++++++++

- long-context-friendly design;
- distinct comparison point relative to Mamba;
- hybrid recurrent/feed-forward character.

Computational complexity
++++++++++++++++++++++++

- Time: recurrence yields linear-time sequence traversal, while channel-mixing
  keeps expressive capacity at each position.
- Memory: more favorable than quadratic attention on long sequences because the
  model aggregates history through recurrent summaries rather than dense token
  pair matrices.
- Best-fit regime: useful for long contexts that benefit from gradual evidence
  accumulation and where attention-map materialization would be wasteful.

Limitations
+++++++++++

- less conventional than BiLSTM or standard Transformer;
- requires careful numerical treatment for stable sequence accumulation.

.. image:: ../img/div.png
