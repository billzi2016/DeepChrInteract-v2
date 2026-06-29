Biological Background
=====================

Switch language: :doc:`../zh/BiologicalBackground`

Enhancer-promoter interaction (EPI) prediction is a sequence-based learning task
centered on distal gene regulation. Enhancers are cis-regulatory DNA elements
that can increase transcription of target genes, often across long genomic
distances. Promoters are proximal regulatory regions near transcription start
sites that recruit the transcriptional machinery. Their coordinated interaction
is one of the main mechanisms through which cell identity, developmental
programs, and disease-associated dysregulation are established.

Why this problem matters
++++++++++++++++++++++++

- Experimental assays such as Hi-C, ChIA-PET, Capture-C, and related 3D genome
  technologies are informative but expensive, slow, and not uniformly available
  across cell types.
- A computational model that infers likely enhancer-promoter pairs from sequence
  can help prioritize regulatory hypotheses, guide follow-up experiments, and
  support interpretation of non-coding variants.
- The project also serves as a benchmark bed for comparing classical CNN-style
  genomic models with newer sequence architectures such as Transformers, Mamba,
  RWKV, mLSTM, and DNA language model encoders.

Biological framing
++++++++++++++++++

In biological terms, the task is not simply motif classification. A practical
EPI predictor must learn several levels of signal:

- local sequence composition, including GC content and low-level nucleotide
  arrangement;
- regulatory motifs associated with transcription factors and chromatin state;
- combinational logic between enhancer and promoter regions;
- long-range context patterns that may be represented better by modern
  sequence models than by shallow local filters alone.

Input and output
++++++++++++++++

The project models each sample as a pair of DNA sequences:

- enhancer sequence;
- promoter sequence.

The output is a binary label:

- ``1`` means the pair is considered interacting;
- ``0`` means the pair is considered non-interacting.

In the current implementation, the core framework supports three input
representations:

- one-hot encoding;
- k-mer token encoding;
- precomputed DNA LLM embeddings.

Project role
++++++++++++

The current repository is a modernization of the original DeepChrInteract code
base. The old Keras/TensorFlow implementation is preserved for historical
reference under ``DeepChrInteract-main(old)/``. The new codebase reimplements
the workflow in PyTorch, removes the PNG-based preprocessing path, expands the
model family to fourteen encoder variants, and standardizes evaluation and
experiment organization.

.. image:: ../img/div.png

