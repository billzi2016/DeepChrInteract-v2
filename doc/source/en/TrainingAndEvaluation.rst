Training and Evaluation
=======================

Switch language: :doc:`../zh/TrainingAndEvaluation`

Training entry point
++++++++++++++++++++

The main entry point is ``python -m src.train``. The current training script
implements a full experiment loop around a configuration object, dataset
construction, model instantiation, validation-driven checkpointing, and JSON
logging.

Core training behavior
++++++++++++++++++++++

The training system includes the following components:

- configuration persistence through ``config.json`` in each experiment folder;
- automatic device selection in the order ``cuda -> mps -> cpu``;
- support for dummy-mode pipeline validation when real genomic data are not yet
  prepared;
- optional k-mer tokenization and LLM embedding loading paths;
- optimizer initialization with Adam;
- optional parameter grouping for DNA LLM fine-tuning;
- cosine annealing warm restarts scheduling;
- gradient clipping at ``max_norm=1.0``;
- early stopping on validation AUROC;
- saving ``best.pt`` and ``last.pt`` checkpoints.

Input pipeline
++++++++++++++

The dataset layer is centered on paired enhancer-promoter input:

- ``EPIDataset`` loads ``seqs_e``, ``seqs_p``, and ``labels`` from NPZ files;
- ``onehot`` mode performs on-the-fly encoding to ``(5, seq_len)`` tensors;
- ``kmer`` mode converts sequences into integer token streams;
- ``llm`` mode reads precomputed embedding vectors from disk;
- ``DummyEPIDataset`` generates synthetic tensors with realistic shapes for
  end-to-end debugging.

Why online encoding matters
+++++++++++++++++++++++++++

One of the main engineering improvements over the legacy code is the removal of
PNG-based intermediates. In the current design:

- sequence strings remain the canonical stored representation;
- one-hot tensors are created in memory only when a batch is loaded;
- unnecessary quantization, image I/O, and temporary storage blow-up are
  avoided.

Validation logic
++++++++++++++++

During training, each epoch is paired with validation:

- loss is computed with ``BCEWithLogitsLoss``;
- validation probabilities are obtained through ``sigmoid``;
- AUROC is used as the primary model-selection criterion;
- early stopping halts training when AUROC no longer improves.

Evaluation outputs
++++++++++++++++++

The evaluation entry point is ``python -m src.evaluate``. It produces:

- AUROC;
- AUPRC;
- F1 score;
- accuracy;
- ROC curve image;
- precision-recall curve image;
- per-seed ``metrics.json``;
- aggregated ``summary.json`` when multiple seeds are summarized.

Practical interpretation
++++++++++++++++++++++++

The metric set is chosen to better reflect the realities of genomic interaction
prediction:

- AUROC measures ranking quality across thresholds;
- AUPRC is especially useful when class imbalance is non-trivial;
- F1 exposes threshold-sensitive balance between precision and recall;
- accuracy is retained for continuity with simpler baseline reporting.

Recommended workflow
++++++++++++++++++++

1. preprocess one cell type into ``train/val/test`` NPZ files;
2. run ``scripts/test_pipeline.py`` if the environment or model stack is new;
3. train a single seed for quick sanity checking;
4. launch a five-seed batch experiment;
5. evaluate and collect per-seed metrics;
6. compare model families and fusion strategies using the saved outputs.

.. image:: ../img/div.png

