Data Organization
=================

	
Data and codes in https://github.com/lichen-lab/DeepChrInteract are listed as follows in order to run ``DeepChrInteract``

- ``DeepChrInteract``

	The ``root`` folder of this project. All files should be stored in this directory and its subdirectories.
	
	- ``DeepChrInteract.py``

		Main file to call all functions. Use ``python3 DeepChrInteract.py -h`` to browse all functions and options

	- ``data_preprocessing.py``
	
		Used to ``preprocess data``, generate ``npz`` and ``png`` files, use ``python3 DeepChrInteract.py -p true -n [file name]``

	- ``model.py``

		Stored ``all models``, including: ``onehot_cnn_one_branch`` / ``onehot_cnn_two_branch`` / ``onehot_embedding_dense`` / ``onehot_embedding_cnn_one_branch`` / ``onehot_embedding_cnn_two_branch`` / ``onehot_dense`` / ``onehot_resnet18`` / ``embedding_cnn_one_branch`` / ``embedding_cnn_two_branch``

	- ``train.py``
	
		Called by ``model.py`` to train the model. Use ``python3 DeepChrInteract.py -m [model name] -t train -n [file name]``

	- ``test.py``

		Called by ``model.py`` to test the model. Use ``python3 DeepChrInteract.py -m [model name] -t test -n [file name]``

	- ``log.txt``

		Used to store prediction results by ``test.py``, Stores the source gene of the timestamp model, the test target gene, the aucroc result, and the Pearson correlation coefficient

	- ``embedding_matrix.npy``
	
		Pretrained ``DNA2VEC embedding`` matrix from ``hg19 human genome``, which contains 4097*100 dimensions (``6mer``, 2**6=4096, where the first line is the initial line, all 0)
	
	- ``resnet.py`` 
		
		Include resnet18, resnet34, resnet50, resnet101, resnet152. This file is a resnet library file.
		
		
	
	- File path ``data``
	
		Store DNA sequences from labelled chromatin interactions and non-chromatin interactions
	
		- File path ``Example: AD2.po``
				
			- ``seq.anchor1.pos.txt`` 
				DNA sequence for chromatin-interacted region1. Each row is a sequence.
				
			- ``seq.anchor2.pos.txt`` 
				DNA sequence for chromatin-interacted region2. Each row is a sequence.

			- ``seq.anchor1.neg2.txt`` 
				DNA sequence for non-chromatin-interacted region1. Each row is a sequence.

			- ``seq.anchor2.neg2.txt`` 
				DNA sequence for non-chromatin-interacted region2. Each row is a sequence.
		
			
		
	- File path ``h5_weights``
		
		Saved weights for neural network
	
	- File path ``result``

		Consists of folders for multiple datasets and multiple model folders for each dataset



.. image:: img/div.png