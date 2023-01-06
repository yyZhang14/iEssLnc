# iEssLnc: Quantitative estimation of lncRNA gene essentialities with meta-path-guided random walks on the lncRNA-protein interaction networks

In this work, we proposed a method to identify essential lncRNAs by taking full advantage of the topological feature of the lncRNA-protein heterogeneous network and lncRNA sequence information. We used lncRNA sequence information to select optimal non-essential lncRNAs and introduced the metapath2vec algorithm to learn low-dimensional embedding representation for lncRNAs in the lncRNA-protein heterogeneous network. We named this method as **iEssLnc**. 


## 1. File descriptions

### 1.1 data

**human**

   - eng.csv: Minimum free energy of secondary structure of sequences.
   - essential.csv : 154 human essential lncRNAs in iEssLnc method.
   - id_lncRNA.txt:  The ID number of lncRNA. In order to distinguish it from protein, lncRNA name needs to be prefixed with "l".
   - id_lncRNA-protein.txt:  The ID number of lncRNA-protein interaction.  In the form of: lncRNA ID_protein ID.
   - id_protein.txt: The ID number of protein. In order to distinguish it from lncRNA, protein name needs to be prefixed with "p".
   - LPI.csv : LncRNA-protein interaction (LPI) data.
   - ncName_ncID_transID.csv: Name, ID, and transcript ID of lncRNAs.
   - transcripts_seq.fasta: Transcriptional sequence of lncRNAs.

**mouse**

   - eng.csv: Minimum free energy of secondary structure of sequences.
   - essential.csv : 25 mouse essential lncRNAs in iEssLnc method.
   - id_lncRNA.txt:  The ID number of lncRNA. In order to distinguish it from protein, lncRNA name needs to be prefixed with "l".
   - id_lncRNA-protein.txt:  The ID number of lncRNA-protein interaction.  In the form of: lncRNA ID_protein ID.
   - id_protein.txt: The ID number of protein. In order to distinguish it from lncRNA, protein name needs to be prefixed with "p".
   - LPI.csv : LncRNA-protein interaction (LPI) data.
   - ncName_ncID_transID.csv: Name, ID, and transcript ID of lncRNAs.
   - transcripts_seq.fasta: Transcriptional sequence of lncRNAs.

### 1.2 metapath

- genMetaPaths.py

```bash
./genMetaPaths. py 400 100 ../data/mouse in_mouse_400_100.txt

# ./ means to run the python file
# 400 indicates that each node generates 400 paths
# 100 indicates that the path length is 100
# .. /data/mouse indicates the path where the data file is stored
# in_mouse_400_100.txt indicates the file name
```

- code_metapath2vec/metapath2vec.cpp

```
1. compile 
cmd "make" in folder code_metapath2vec/ 

2. run "metapath2vec" in folder code_metapath2vec/ 
./metapath2vec -train ../../data/mouse/in_mouse_400_100.txt -output ../../result/mouse/out_mouse_400_100 -pp 1 -size 64 -window 7 -negative 5 -threads 16 

3. parameter 
Input:
[Required options]
● ./metapath2vec : run cpp file
● -train <file> : Use text data from <file> to train the model, e.g., ../../data/mouse/in_mouse_300_40.txt
● -output <file>:  Use <file> to save the resulting word vectors , e.g., ../../result/mouse/out_mouse_300_40
● -pp <int>: Use metapath2vec++ or metapath2vec; default is 1 (metapath2vec++); for metapath2vec, use 0
● -size <int>: Set size of word vectors; default is 100
● -window <int>: Set max skip length between words; default is 5
● -negative <int>: Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
● -threads <int>: Use <int> threads (default 12)

[Optional options]
-sample <float>
Set threshold for occurrence of words. Those that appear with higher frequency in the training data
will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
-iter <int>
Run more training iterations (default 5)
-min-count <int>
This will discard words that appear less than <int> times; default is 5
-alpha <float>
Set the starting learning rate; default is 0.025 for skip-gram
-classes <int>
Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
-debug <int>
Set the debug mode (default = 2 = more info during training)
-save-vocab <file>
The vocabulary will be saved to <file>
-read-vocab <file>
The vocabulary will be read from <file>, not constructed from the training data

Output:
one vector file for each node in binary format 		(e.g., out_mouse_400_100)
the same vector file for each node in text format 	(e.g., out_mouse_400_100.txt)
```

### 1.3 model

- **figure.py** : Plot ROC and PR curves of the SVM,RF and MLP model.
- **metrics.py**: Compute the predicted performance indicators.
- **mlp.py**: The MLP model code , which can compute the performance of the dataset , also can be used select the optimal parameter for MLP.
- **score.py**: Use the MLP model to compute the predicted scores of all lncRNAs.
- **rf.py**: The RF model code.
- **shuffle_mlp.py**: The MLP model was used for Shuffle experiment. The dataset was fixed, and the positive and negative labels were shuffled 1000 times. Each performance was calculated separately.
- **shuffle_rf.py**: The RF model was used for Shuffle experiment. The dataset was fixed, and the positive and negative labels were shuffled 1000 times. Each performance was calculated separately.
- **svm.py**: The SVM model, which is used for computing the performance of human and mouse dataset , and doing shuffle experiment 1000 times as described above.
- **svm_module.py** : The SVM model code.

### 1.4 pre

- **cal_gic.py**: In this file, we run the code using files `ncName_ncID_transID.csv`, `eng.csv `and `transcripts_seq.fasta`, which are stored in the `data/human` or  `data/mouse`  folder. The result from running the code are stored in the `result/human` and `result/mouse` folder, named `GIC_score.csv`.

- **idmap.py**: In this file, we run the code using `LPI.csv`, which are stored in `data/human` or  `data/mouse` folder.  This code generates three files,` id_lncRNA.txt`, `id_protein.txt` and `id_lncRNA_protein.txt` .  The results are stored in` data/human` and  `data/mouse`folder. 

- **pro_path.py**: In this file, we use the code using `positive.csv` , `negative.csv` , which  are stored in `result/human`or `result/mouse` folder.  This code also uses text file generated by `metapath2vec.cpp`, e.g., `out_mouse_400_100.txt` are stored in `result/mouse/para/400_100_64`. This code generates  files `dataset1.csv`, e.g. these files are stored in  `result/mouse/para/400_100_64`.

### 1.5 result

- **human**

  In this folder, `positive.csv`  stores 154 human essential lncRNAs(positive samples). `negative.csv` stores negative samples of which the quantity equal s to the number of positive samples(according to "GIC" negative sample strategy ).  `GIC_score.csv` stores the GIC score of all lncRNA genes. 
  In the `human/para` folder, `300_30_256`  is the optimal parameter of metapath2vec in human dataset. 300 is the number of walks per node, 30 is the length of each walk , and 256 is the dimension of embedding representation.  

  In the `300_30_256` folder, `alldata.csv` file stores all lncRNA feature vector, which are generated by the metapath2vec algorithm. `out_human_300_30.txt` are the result of `metapath2vec.cpp` . `dataset1.csv` file stores positive  and negative samples feature vector.  `MLP_pred.csv` , `SVM_pred.csv`  and `RF_pred.csv` add predicted label ,respectively.

- **mouse**

  In this folder, all files have the same distribution as the `result/human` folder, but this folder works for Mouse.  It should be noted that there are 25 essential lncRNAs in mouse dataset.

- **GIC select** 

  This folder contains the `human` and `mouse` folders. `unlabel.csv` file is all unlabelled lncRNAs with GIC score.  Take a random 1000 negative samples from unlabelled data , as many as positive samples each time. The negative samples taken each time are stored in file `human_random.csv`, and the performance indicators are stored in file `human_random_metrics.csv`(use SVM model).

  In the `300_30_256` folder, the negative samples are selected from the top, the bottom and around three median quartiles (25%, 50% and 75% percentiles) of the sorted lncRNA gene list. The positive samples and different samples are stored in dataset1.csv (the top), dataset2.csv ( 25%), dataset3.csv(50%), dataset4.csv(75%), dataset5.csv(the bottom) .  And `performance.csv` file stores performance metrics for the five different datasets.

  In the `GIC select/mouse` folder, all files have the same distribution as the `GIC select/human` folder.

- **shuffle** 

  This folder contains two folders, `300_30_256` for human data and `400_100_64` for mouse data. The dataset was fixed, and the positive and negative labels were shuffled 1000 times.  The shuffled label stores in `dataset1.csv` .   And `human/mouse_shuffle_MLP.csv` , `human/mouse_shuffle_RF.csv1`  and `human/mouse_shuffle_SVM.csv` are prediction performance for every condition and different model.

- **SVM_score、MLP_score and RF_score** 

  These folders record the predicted scores of each model for all human and mouse lncRNAs.

- **POCPR**

  These data are used to plot ROC curves and PR curves. The values of fpr and tpr under each threshold were recorded.

---

## 2. How to use iEssLnc

The following steps take human as an example.

### 2.1 Construct lncRNA-protein heterogeneous network

Step1: Users can run `idmap.py`, input `LPI.csv`  , output `id_lncRNA.txt`, `id_protein.txt` and `id_lncRNA_protein.txt`.

### 2.2 Heterogeneous representation learning

Step1:  On linux environment, Users can use `genMetaPaths.py` to generate the path file , e.g. `in_human_300_30.txt`. Users need enter 

```bash
./genMetaPaths.py 300 30 ../data/mouse in_human_300_30.txt
```

Step2: In `code_metapath2vec` folder, users firstly  enter `make` to compile `metapath2vec.cpp` , then enter the following code to run cpp file.

```bash
./metapath2vec -train ../../data/human/in_human_300_30.txt -output ../../result/human/out_human_300_30 -pp 1 -size 256 -window 7 -negative 5 -threads 16
```

Users can get `out_human_300_30.txt `as output file, then move this file to Windows environment using XFTP software, store it in the `result/human/para/300_30_256 folder.

### 2.3 Select optimal non-essential lncRNAs

Step1:  Users can run `cal_gic.py` to get GIC score of all lncRNAs, input `ncName_ncID_transID.csv`, `eng.csv `and `transcripts_seq.fasta`, output `GIC_score.csv`.
Step2: In `GIC_score.csv` file, the value is sorted in ascending order, and the data with the same  number of positive samples are respectively selected to form `negative.csv` , which are stored in `result/human` folder. 
Step3:  User can run `pro_path.py`, process the `out_human_300_30.txt` file and select the lncRNA vectors in the `positive.csv`, `negative.csv ` files respectively to form `dataset1.csv` which are stored in `result/human/para/300_30_256`folder.

### 2.4 Supervised machine learning

#### 2.4.1 SVM model

| dataset |  C   | gamma | kernel function |
| :-----: | :--: | :---: | :-------------: |
|  Human  |  1   |  0.1  |       rbf       |
|  Mouse  | 100  | 0.001 |       rbf       |

The table above shows the parameters used by the SVM model under different datasets.
For example, users can run `svm.py`, input `dataset1.csv`, use leave-one-out cross-validation, and get  accuracy, precision and other 's performance indicators.
#### 2.4.2 RF model

| dataset | n_estimators | max_depth |
| :-----: | :----------: | :-------: |
|  Human  |     100      |     3     |
|  Mouse  |      50      |     3     |

User can run `rf.py` 

#### 2.4.3 MLP model

| dataset | The number of neurons in hidden layer | max_iter |
| :-----: | :-----------------------------------: | :------: |
|  Human  |                  64                   |   200    |
|  Mouse  |                   4                   |   200    |

User can run `mlp.py`.

