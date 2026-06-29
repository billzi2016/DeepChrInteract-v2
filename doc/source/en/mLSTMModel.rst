mLSTM Model
===========

Switch language: :doc:`../zh/mLSTMModel`

``M5`` implements the matrix-memory LSTM route inspired by xLSTM-style design.

What makes mLSTM different
++++++++++++++++++++++++++

Unlike a conventional LSTM that stores vector-valued hidden memory, mLSTM uses a
matrix-style memory update. In this repository, the design is motivated by the
idea that richer recurrent state may capture more structured sequence
dependencies than standard recurrent cells.

Core idea
+++++++++

- query, key, and value projections are formed at each sequence step;
- memory is updated as a matrix rather than a simple vector state;
- gating is stabilized in log space;
- a bidirectional wrapper is used so the encoder sees context from both sides.

Why it is interesting for genomic sequences
+++++++++++++++++++++++++++++++++++++++++++

Genomic regulatory sequences often involve combinational dependencies rather
than isolated motifs. Matrix-memory recurrence is attractive because it can, in
principle, preserve richer interaction structure across positions than a simple
hidden vector.

Strengths
+++++++++

- richer recurrent state than vanilla LSTM;
- explicit recurrence without relying on quadratic attention;
- useful as a modern recurrent alternative in long-sequence benchmarking.

Tradeoffs
+++++++++

- more specialized and harder to reason about than plain BiLSTM;
- still sequential in spirit, even if conceptually more expressive;
- more implementation complexity than simpler baselines.

Project role
++++++++++++

``M5`` is the project's advanced recurrent representative. It helps position the
benchmark beyond classical LSTMs and gives the documentation a bridge between
legacy sequence models and newer state-space or attention families.

.. image:: ../img/div.png

