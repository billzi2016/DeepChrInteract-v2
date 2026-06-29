Data Organization
=================

Switch language: :doc:`../zh/DataOrganization`

Data and codes in the original DeepChrInteract project were organized around a
single entry script, preprocessing helpers, model definitions, training,
testing, logged results, and sequence resources.

The current project keeps the same overall scientific workflow but restructures
the repository for maintainability.

Repository structure
++++++++++++++++++++

.. code-block:: text

   .
   ├── src/
   │   ├── config.py
   │   ├── dataset.py
   │   ├── encoders.py
   │   ├── train.py
   │   ├── evaluate.py
   │   └── models/
   ├── scripts/
   │   ├── preprocess.py
   │   ├── run_experiment.sh
   │   └── test_pipeline.py
   ├── results/
   ├── latex/
   ├── DeepChrInteract-main(old)/
   ├── README.md
   ├── PRD.md
   └── TASK.md

Raw data layout
+++++++++++++++

The new preprocessing entry expects four text files per cell type:

.. code-block:: text

   data/raw/{cell_type}/
       seq.anchor1.pos.txt
       seq.anchor2.pos.txt
       seq.anchor1.neg.txt
       seq.anchor2.neg.txt

Processed data layout
+++++++++++++++++++++

After preprocessing, the repository stores split datasets as:

.. code-block:: text

   data/{cell_type}/
       train.npz
       val.npz
       test.npz

Each NPZ file stores:

- ``seqs_e``: enhancer sequence strings;
- ``seqs_p``: promoter sequence strings;
- ``labels``: binary interaction labels.

Results layout
++++++++++++++

.. code-block:: text

   results/{exp_id}/seed{n}/
       config.json
       best.pt
       last.pt
       history.json
       metrics.json
       roc_curve.png
       pr_curve.png

Legacy archive
++++++++++++++

The original Keras-based implementation remains under
``DeepChrInteract-main(old)/``. It is preserved for historical comparison,
method tracing, and reference to the original writing and diagrams.

.. image:: ../img/div.png

