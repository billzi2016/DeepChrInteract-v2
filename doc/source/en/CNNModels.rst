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

At the operator level, a 1D convolution behaves like a trainable motif scanner:

.. math::

   h_t = \sigma\!\left(\sum_{c=1}^{C}\sum_{i=0}^{k-1} w_{c,i}\,x_{c,t+i} + b\right)

where :math:`x` is the multi-channel sequence input, :math:`k` is the kernel
width, and :math:`h_t` is the activation at position :math:`t`. Each filter can
be interpreted as a learned detector for short sequence patterns and their
local variants.

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

Because it avoids pair fusion and avoids recurrent or attention mechanisms, M1
is the cleanest answer to a foundational question: how far can local feature
detectors and hierarchical pooling go on their own?

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

Its computation can be summarized as:

.. math::

   h_e = f_e(x_e), \qquad h_p = f_p(x_p), \qquad z = \mathrm{Fuse}(h_e, h_p)

This makes the architecture explicitly separate the two encoding problems
before asking a fusion layer to model compatibility, asymmetry, or synergy.

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

Formally, the input representation becomes:

.. math::

   s = (t_1, \dots, t_n), \qquad e_i = E[t_i]

where :math:`t_i` is a k-mer token and :math:`E` is the embedding table. The
CNN then operates over token vectors rather than raw nucleotide channels, which
places M3 between classical motif CNNs and language-model style representations.

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
