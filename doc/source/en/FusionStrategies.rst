Fusion Strategies
=================

Switch language: :doc:`../zh/FusionStrategies`

Enhancer-promoter prediction is not only about single-sequence encoding. It is
also about how two encoded regions are combined before classification.

Supported strategies
++++++++++++++++++++

- ``concat``
- ``add``
- ``subtract``
- ``multiply``
- ``bilinear``
- ``concat_sub_mul``

If enhancer and promoter embeddings are written as :math:`h_e` and :math:`h_p`,
the main fusion forms can be summarized as:

.. math::

   \mathrm{concat}: [h_e; h_p]

.. math::

   \mathrm{add}: h_e + h_p

.. math::

   \mathrm{subtract}: h_e - h_p

.. math::

   \mathrm{multiply}: h_e \odot h_p

.. math::

   \mathrm{concat\_sub\_mul}: [h_e; h_p; h_e - h_p; h_e \odot h_p]

Why fusion matters
++++++++++++++++++

Even a strong encoder can underperform if the pairwise interaction layer is too
weak. Fusion determines how much relational information between enhancer and
promoter embeddings is exposed to the classifier.

This is why fusion should not be treated as a minor implementation detail. It
defines what kinds of pair structure the classifier is allowed to see directly.

Interpretation of the main options
++++++++++++++++++++++++++++++++++

- ``concat`` keeps the two vectors intact and lets the classifier learn the
  interaction.
- ``add`` emphasizes shared magnitude and symmetric aggregation.
- ``subtract`` exposes directional difference.
- ``multiply`` highlights element-wise agreement or co-activation.
- ``bilinear`` offers a more expressive learned pairwise interaction.
- ``concat_sub_mul`` explicitly combines identity, difference, and agreement
  signals in one representation.

Default choice
++++++++++++++

The repository uses ``concat_sub_mul`` as the default because it tends to be a
balanced representation-rich option without requiring the parameter overhead of a
full bilinear interaction.

.. image:: ../img/div.png
