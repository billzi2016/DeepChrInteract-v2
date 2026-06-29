ResNet Models
=============

Switch language: :doc:`../zh/ResNetModels`

Residual convolutional models should remain part of the documented model system.
They are not just historical leftovers. They represent the natural deep-CNN
expansion path for this project and are especially relevant now that hardware
constraints are less restrictive.

Why ResNet belongs here
+++++++++++++++++++++++

The project already documents simple CNN baselines. A ResNet family extends that
line in the most classical way:

- keep convolution as the main local-pattern engine;
- deepen the network substantially;
- use skip connections to stabilize optimization and preserve gradient flow.

Relationship to earlier resource limits
+++++++++++++++++++++++++++++++++++++++

In earlier environments, GPU memory constraints could make deeper convolutional
models less convenient to prioritize. In a small-memory setup, lighter CNN
variants are often the more practical first choice.

With substantially larger modern GPU memory budgets, that constraint is much
less binding. This makes residual CNNs worth restoring as an active evaluation
route rather than leaving them only as archived diagrams.

ResNet18 and ResNet34 in context
++++++++++++++++++++++++++++++++

The legacy documentation already referenced residual architectures such as
ResNet18 and ResNet34. In the current documentation, they should be understood
as:

- deeper convolutional alternatives to the plain CNN baselines;
- a bridge between historical local-feature models and later hybrid or
  Transformer-style systems;
- an architecture family that is now practical to include again in comparative
  experiments.

What residual connections add
+++++++++++++++++++++++++++++

Compared with a plain CNN stack, a residual network offers:

- easier optimization at greater depth;
- stronger hierarchical feature extraction;
- better preservation of lower-level features while building higher-level
  abstractions.

Why ResNet still matters for genomic sequence modeling
++++++++++++++++++++++++++++++++++++++++++++++++++++++

For regulatory DNA tasks, deeper convolution can be useful when the aim is to:

- capture layered motif compositions;
- aggregate local patterns into broader sequence features;
- stay within the convolutional inductive-bias family without jumping directly
  to attention or state-space formalisms.

Documentation role
++++++++++++++++++

This page keeps ResNet explicitly inside the model map. Even if the current
active registry is centered on ``M1-M14``, ResNet should be treated as a
documented, expected branch of the project rather than an obsolete footnote.

.. image:: ../img/div.png

