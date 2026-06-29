快速开始
========

切换语言: :doc:`../en/QuickStart`

DeepChrInteract 原始动机
++++++++++++++++++++++++

旧版文档强调：虽然基于染色质相互作用区域侧翼 DNA 序列的深度学习方法不断发展，
但一个能够系统整合并比较不同深度学习架构的工具包仍然不足。

当前项目保留这一目标，并将其扩展为一个基于 PyTorch 的增强子-启动子相互作用
预测与模型比较框架。

环境要求
++++++++

旧版文档给出的基础环境如下：

- CPU 内存建议 ``16GB``
- GPU 显存建议 ``8GB``
- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0

当前项目实际依赖已经切换为：

- Python 3.10+；
- PyTorch 2.x；
- numpy、scikit-learn、matplotlib、tqdm；
- transformers；
- 可选 ``mamba-ssm``。

安装
++++

.. code:: bash

   git clone <your-repository-url>
   cd Enhancer-Promoter-Interaction
   pip install -r requirements.txt

如果要使用 Mamba，可额外安装：

.. code:: bash

   pip install mamba-ssm

数据预处理
++++++++++

当前版本直接读取原始序列文本文件，并输出 ``train.npz``、``val.npz``、
``test.npz``，不再生成 PNG 中间文件。

.. code:: bash

   python scripts/preprocess.py \
       --raw_dir data/raw \
       --cell_type GM12878 \
       --out_dir data

无真实数据时的完整管道测试
++++++++++++++++++++++++++++

.. code:: bash

   python scripts/test_pipeline.py
   python scripts/test_pipeline.py --quick

单实验训练
++++++++++

.. code:: bash

   python -m src.train \
       --model_id M2 \
       --exp_id E03 \
       --encoding_mode onehot \
       --fusion_strategy concat_sub_mul \
       --cell_type GM12878 \
       --seed 0

单实验评估
++++++++++

.. code:: bash

   python -m src.evaluate \
       --model_id M2 \
       --exp_id E03 \
       --encoding_mode onehot \
       --cell_type GM12878 \
       --seed 0

五个随机种子的批量实验
++++++++++++++++++++++++

.. code:: bash

   bash scripts/run_experiment.sh E03 M2 GM12878 onehot concat_sub_mul

DNA 大语言模型流程
++++++++++++++++++

``M13`` 支持先离线生成 embedding，再重复训练：

.. code:: bash

   python -c "
   from src.encoders import LLMEncoder
   enc = LLMEncoder('dnabert2')
   # 从处理后的数据中读取 enhancer / promoter 序列后调用 encode_dataset()
   "

MAE 预训练流程
+++++++++++++++

.. code:: bash

   python -m src.train --model_id M14 --exp_id E16 --pretrain
   python -m src.train --model_id M14 --exp_id E16

文档发布
++++++++

本项目文档将采用 Sphinx 构建静态 HTML，再发布到 GitHub Pages。

.. image:: ../img/div.png

