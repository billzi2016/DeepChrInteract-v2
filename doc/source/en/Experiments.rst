Experiments
===========

Switch language: :doc:`../zh/Experiments`

Experiment philosophy
+++++++++++++++++++++

The project is designed as a structured benchmark rather than a single-model
implementation. The experiment board in ``TASK.md`` organizes work into a model
matrix, fusion ablations, and cross-cell-type generalization analysis.

Primary experiment matrix
+++++++++++++++++++++++++

The current plan tracks experiments ``E01`` through ``E16``:

- ``E01``: M1 CNN single-branch baseline
- ``E02``: M2 CNN dual-branch baseline
- ``E03``: M2 with ``concat_sub_mul`` fusion
- ``E04``: M3 k-mer + CNN
- ``E05``: M4 BiLSTM
- ``E06``: M6 standard Transformer
- ``E07``: M11 CNN + BiLSTM
- ``E08``: M12 CNN + Transformer
- ``E09``: M13 DNABERT-2 frozen
- ``E10``: M13 HyenaDNA fine-tune
- ``E11``: M9 Mamba
- ``E12``: M7 Linear Transformer
- ``E13``: M8 iTransformer
- ``E14``: M10 RWKV
- ``E15``: M5 mLSTM
- ``E16``: M14 MAE pretrain + fine-tune

Fusion ablation
+++++++++++++++

The fusion strategy itself is a research axis. The documented ablation plan
includes:

- ``F01``: concat
- ``F02``: add
- ``F03``: subtract
- ``F04``: multiply
- ``F05``: bilinear
- ``F06``: concat_sub_mul

Why this comparison matters
+++++++++++++++++++++++++++

Enhancer-promoter prediction is naturally a pairwise learning problem. A strong
encoder can still underperform if the interaction representation is weak. By
explicitly exposing multiple fusion operators, the project separates two
questions:

- how expressive the sequence encoder is;
- how effectively enhancer and promoter features are combined.

Seed strategy and reproducibility
+++++++++++++++++++++++++++++++++

Each experiment is intended to run on five random seeds:

- seed ``0``
- seed ``1``
- seed ``2``
- seed ``3``
- seed ``4``

This reduces the risk of over-interpreting a favorable or unfavorable single
run and gives a more stable estimate of expected behavior.

Expected outputs per experiment
+++++++++++++++++++++++++++++++

Every seed directory stores:

- configuration snapshot;
- checkpoints;
- training history;
- test metrics;
- ROC curve;
- PR curve.

At the experiment level, summary statistics are intended to report mean and
standard deviation across seeds.

Showcase value
++++++++++++++

This experiment design demonstrates more than model implementation. It shows:

- baseline preservation;
- controlled architectural expansion;
- reproducible training organization;
- explicit ablation thinking;
- practical readiness for comparative sequence modeling studies.

.. image:: ../img/div.png

