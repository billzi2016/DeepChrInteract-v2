MAE Pretraining Model
=====================

Switch language: :doc:`../zh/MAEModel`

``M14`` is the self-supervised pretraining route of the repository.

Core idea
+++++++++

The model uses a Masked Autoencoder style workflow:

1. mask a large fraction of sequence patches;
2. encode the visible subset;
3. reconstruct the masked content;
4. reuse the learned encoder for supervised EPI prediction.

The reconstruction objective can be written as:

.. math::

   \mathcal{L}_{\mathrm{MAE}} =
   \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}}
   \ell(\hat{x}_i, x_i)

where :math:`\mathcal{M}` is the masked patch set and :math:`\ell` measures how
well the decoder recovers hidden sequence content.

Why this is different from M13
+++++++++++++++++++++++++++++++

``M13`` imports external pretrained genomic knowledge.
``M14`` learns a task-adjacent representation directly from the project's own
data distribution through self-supervision.

Why this matters
++++++++++++++++

For regulatory genomics, labels are valuable and often limited. A self-supervised
route is attractive because it can:

- extract structure from unlabeled or weakly labeled sequence data;
- adapt the representation to the local data distribution;
- provide a middle path between scratch training and large external foundation
  models.

Implementation logic
++++++++++++++++++++

- Encoder: Transformer-style sequence encoder
- Pretraining objective: masked reconstruction
- Finetuning objective: paired enhancer-promoter classification

This means the encoder is first optimized to model sequence structure without
labels, then repurposed as a supervised feature extractor for the downstream
interaction task.

Project role
++++++++++++

This model gives the documentation a full representation-learning ladder:

- scratch baselines;
- advanced sequence architectures;
- external foundation models;
- in-project self-supervised pretraining.

Computational complexity
++++++++++++++++++++++++

- Time: pretraining is substantially more expensive than direct supervised
  training because reconstruction must be learned before downstream finetuning.
- Memory: encoder-decoder pretraining and later finetuning make this route
  heavier than a simple task-only baseline.
- Best-fit regime: appropriate when unlabeled sequence volume is available and a
  richer in-project representation is worth extra training cost.

.. image:: ../img/div.png
