Quick Start
===========

Switch language: :doc:`../zh/QuickStart`

Introduction to DeepChrInteract
+++++++++++++++++++++++++++++++

Though deep learning methods have been widely developed for predicting chromatin
interactions using flanking DNA sequence in identified chromatin interaction
regions, a comprehensive software toolkit to integrate and evaluate different
deep learning architectures are under-developed.

The modern project keeps that original motivation and extends it into a
PyTorch-based benchmark and research framework for enhancer-promoter
interaction prediction.

System requirements
+++++++++++++++++++

The original documentation listed the following baseline environment:

- CPU memory is recommended as ``16GB``
- GPU memory is recommended as ``8GB``
- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0

The current repository uses a different runtime:

- Python 3.10+ is recommended;
- PyTorch 2.x;
- numpy, scikit-learn, matplotlib, tqdm;
- transformers for DNA language model backbones;
- optional ``mamba-ssm`` in CUDA environments.

Installation
++++++++++++

Clone the project and install dependencies:

.. code:: bash

   git clone <your-repository-url>
   cd Enhancer-Promoter-Interaction
   pip install -r requirements.txt

Optional dependency for the Mamba model:

.. code:: bash

   pip install mamba-ssm

Data preprocessing
++++++++++++++++++

The current pipeline consumes raw text sequence files and converts them into
``train.npz``, ``val.npz``, and ``test.npz`` splits without generating PNG
intermediates.

.. code:: bash

   python scripts/preprocess.py \
       --raw_dir data/raw \
       --cell_type GM12878 \
       --out_dir data

Pipeline validation without real data
+++++++++++++++++++++++++++++++++++++

The repository includes a dummy-mode validation path for testing the training
and evaluation pipeline before real biological data are available.

.. code:: bash

   python scripts/test_pipeline.py
   python scripts/test_pipeline.py --quick

Single experiment training
++++++++++++++++++++++++++

.. code:: bash

   python -m src.train \
       --model_id M2 \
       --exp_id E03 \
       --encoding_mode onehot \
       --fusion_strategy concat_sub_mul \
       --cell_type GM12878 \
       --seed 0

Evaluation
++++++++++

.. code:: bash

   python -m src.evaluate \
       --model_id M2 \
       --exp_id E03 \
       --encoding_mode onehot \
       --cell_type GM12878 \
       --seed 0

Five-seed batch experiment
++++++++++++++++++++++++++

.. code:: bash

   bash scripts/run_experiment.sh E03 M2 GM12878 onehot concat_sub_mul

DNA language model workflow
+++++++++++++++++++++++++++

For ``M13``, embeddings can be precomputed once and reused:

.. code:: bash

   python -c "
   from src.encoders import LLMEncoder
   enc = LLMEncoder('dnabert2')
   # Load enhancer/promoter sequences from processed data and call encode_dataset()
   "

MAE pretraining workflow
++++++++++++++++++++++++

.. code:: bash

   python -m src.train --model_id M14 --exp_id E16 --pretrain
   python -m src.train --model_id M14 --exp_id E16

Documentation deployment
++++++++++++++++++++++++

This project is intended to be published as a static documentation site through
GitHub Pages after Sphinx builds the HTML output.

.. image:: ../img/div.png
