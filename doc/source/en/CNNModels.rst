CNN Models
==========

Switch language: :doc:`../zh/CNNModels`

This page covers the convolutional baseline family: ``M1``, ``M2``, and ``M3``.
These models are the most direct continuation of the original DeepChrInteract
design philosophy and remain important because they anchor the rest of the
benchmark in interpretable local-pattern learning.

Why CNNs matter for EPI
+++++++++++++++++++++++

DNA sequence prediction often starts with local motif detection. Convolutional
layers are naturally suited for:

- scanning the sequence for short recurring nucleotide patterns;
- building motif combinations through deeper layers;
- offering relatively stable optimization and efficient training;
- providing strong baseline performance without requiring external pretraining.

M1: CNN single-branch baseline
++++++++++++++++++++++++++++++

``M1`` uses a single convolutional branch and acts as the simplest baseline in
the project.

- Input: one-hot encoded sequence tensor ``(B, 5, L)``
- Backbone: two-stage ``Conv1d -> BN -> ReLU -> MaxPool`` stack
- Head: global average pooling followed by fully connected layers
- Role: minimal local-feature baseline

Interpretation:

- Useful when the goal is to test whether local sequence composition alone can
  already produce non-trivial predictive signal.
- It is intentionally simple and easy to compare against more expressive models.

M2: CNN dual-branch baseline
++++++++++++++++++++++++++++

``M2`` extends the convolutional baseline to a paired-input setting.

- Input: enhancer and promoter are encoded separately
- Backbone: the same convolutional encoder is applied to both branches
- Fusion: branch outputs are combined through the project fusion module
- Role: stronger reproduction of the original paired-region setup

Why it matters:

- EPI is a pairwise problem, not a single-sequence classification task.
- Separate branches let the model preserve region-specific representations
  before interaction modeling.

M3: k-mer embedding plus CNN
++++++++++++++++++++++++++++

``M3`` changes the input representation rather than the high-level convolutional
idea.

- Input: k-mer token sequence
- Embedding: learnable token embedding table
- Encoder: four convolutional blocks over embedding channels
- Role: bridge between symbolic tokenization and convolutional feature learning

Why use k-mers:

- They expose local compositional units larger than a single nucleotide.
- They can sometimes make biologically meaningful subsequence patterns easier to
  capture than raw one-hot channels alone.

Shared strengths
++++++++++++++++

- Efficient and stable training
- Strong local pattern extraction
- Clear baseline value for benchmarking
- Easy interpretability compared with heavier architectures

Shared limitations
++++++++++++++++++

- Pure CNNs may struggle to represent long-range dependencies as naturally as
  recurrent, attention-based, or state-space models.
- Their receptive field grows with depth and stride design rather than through
  explicit sequence-wide interaction.

Relationship to the legacy project
++++++++++++++++++++++++++++++++++

The CNN family is the closest direct descendant of the original DeepChrInteract
work. For that reason, it plays two roles at once:

- historical continuity with the original project;
- practical baseline against which all later architectures are evaluated.

.. image:: ../img/div.png

