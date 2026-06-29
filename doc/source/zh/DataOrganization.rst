数据与目录组织
==============

切换语言: :doc:`../en/DataOrganization`

原始 DeepChrInteract 项目围绕主入口脚本、预处理脚本、模型定义、训练测试逻辑、
日志结果以及 DNA 序列资源组织代码与数据。

当前项目保留相同的科学工作流，但用更易维护的目录结构重组。

仓库结构
++++++++

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

原始数据布局
++++++++++++

新的预处理入口默认每个细胞系读取四个文本文件：

.. code-block:: text

   data/raw/{cell_type}/
       seq.anchor1.pos.txt
       seq.anchor2.pos.txt
       seq.anchor1.neg.txt
       seq.anchor2.neg.txt

处理后数据布局
++++++++++++++

预处理后保存为：

.. code-block:: text

   data/{cell_type}/
       train.npz
       val.npz
       test.npz

每个 NPZ 包含：

- ``seqs_e``：增强子序列字符串；
- ``seqs_p``：启动子序列字符串；
- ``labels``：二分类标签。

结果目录布局
++++++++++++

.. code-block:: text

   results/{exp_id}/seed{n}/
       config.json
       best.pt
       last.pt
       history.json
       metrics.json
       roc_curve.png
       pr_curve.png

旧版归档
+++++++++++++++

原始 Keras 实现仍保留在 ``DeepChrInteract-main(old)/`` 中，用于历史比对、
方法溯源以及查看旧版图示与文档写法。

.. image:: ../img/div.png
