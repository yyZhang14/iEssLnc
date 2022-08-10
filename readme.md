# PredEssLnc: Identification and prediction of essential lncRNA genes based on heterogeneous network embedding representation.

---



In this work, we proposed a method to identify essential lncRNAs by taking full advantage of the topological feature of the lncRNA-protein heterogeneous network and lncRNA sequence information. We used lncRNA sequence information to select reliable negative samples and introduced the metapath2vec algorithm to learn low-dimensional embedding representation for lncRNAs in the lncRNA-protein heterogeneous network. We named this method as PredEssLnc(Predict essential lncRNAs). 

---

## 1. File descriptions
### 1.1 data

- compare
> The "GIC" method and “random”method select negative samples, and compare the performance of different data sets under different strategies.

   - Human.xlsx : 61 human essential lncRNAs in Section 3.4 Comparison with state-of-arts methods
   - human_dataset1.csv : Positive:negative =1:1 in human dataset using "random" strategy.
   - human_dataset2.csv: Positive:negative =1:2 in human dataset using "random" strategy.
   - Mouse.xlsx: 7 mouse essential lncRNAs in Section 3.4 Comparison with state-of-arts methods
   - mouse_dataset1.csv: Positive:negative =1:1 in mouse dataset using "random" strategy.
   - mouse_dataset2.csv: Positive:negative =1:1 in mouse dataset using "random" strategy.
   -  para_random.xlsx:  Compare the performance of different data sets under different strategies.
- human
   - eng.csv: Minimum free energy of secondary structure of sequences.
   - essential.csv : 154 human essential lncRNAs in PredEssLnc method.
   - id_lncRNA.txt:  The ID number of lncRNA. In order to distinguish it from protein, lncRNA name needs to be prefixed with "l".
   - id_lncRNA-protein.txt:  The ID number of lncRNA-protein interaction.  In the form of: lncRNA ID_protein ID.
   - id_protein.txt: The ID number of protein. In order to distinguish it from lncRNA, protein name needs to be prefixed with "p".
   - LPI.csv : LncRNA-protein interaction (LPI) data.
   - ncName_ncID_transID.csv: Name, ID, and transcript ID of lncRNAs.
   - transcripts_seq.fasta: Transcriptional sequence of lncRNAs.
- mouse
   - eng.csv: Minimum free energy of secondary structure of sequences.
   - essential.csv : Mouse essential lncRNAs in PredEssLnc method.
   - id_lncRNA.txt:  The ID number of lncRNA. In order to distinguish it from protein, lncRNA name needs to be prefixed with "l".
   - id_lncRNA-protein.txt:  The ID number of lncRNA-protein interaction.  In the form of: lncRNA ID_protein ID.
   - id_protein.txt: The ID number of protein. In order to distinguish it from lncRNA, protein name needs to be prefixed with "p".
   - LPI.csv : LncRNA-protein interaction (LPI) data.
   - ncName_ncID_transID.csv: Name, ID, and transcript ID of lncRNAs.
   - transcripts_seq.fasta: Transcriptional sequence of lncRNAs.
### 1.2 img
All pictures in paper.
### 1.3 metapath

- genMetaPaths.py
```bash
./genMetaPaths. py 300 40 ../data/mouse in_ mouse_300_40.txt

# ./ means to run the python file
# 300 indicates that each node generates 300 paths
# 40 indicates that the path length is 40
# .. /data/mouse indicates the path where the data file is stored
# in_mouse_300_40.txt indicates the file name
```

- code_metapath2vec/metapath2vec.cpp
```
1. compile 
cmd "make" in folder code_metapath2vec/ 

2. run "metapath2vec" in folder code_metapath2vec/ 
./metapath2vec -train ../../data/mouse/in_mouse_300_40.txt -output ../../result/mouse/out_mouse_300_40 -pp 1 -size 32 -window 7 -negative 5 -threads 32 

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
one vector file for each node in binary format 		(e.g., out_mouse_300_40)
the same vector file for each node in text format 	(e.g., out_mouse_300_40.txt)
```
### 1.4 model
**cnn_human.py**
The CNN  model of human datasets.
**cnn_mouse.py**
The CNN model of mouse datasets.
**compare.py**
In human datasets, select 61 essential lncRNAs as test set, remaining lncRNAs in H1 and H2 datasets as train set respectively.
In mouse datasets, select 7 essential lncRNA as test set, remaining lncRNAs in M1 and M2 datasets as train set respectively.
**figure.py**
Plot  ROC curves of SVM, CNN and RF models in H1 , H2 , M1 and M2 datasets.
**para_figure.py**
Plot figure ： the performance comparison of the metapath2vec algorithm under different parameters in different data sets.
**para_svm.py**
Plot figure: The performance comparison of SVM model under different parameters in different data sets.
**rf.py**
The RF model of human and mouse datasets.
**svm.py**
The SVM model of human and mouse datasets.
### 1.5 pre

- cal_gic.py

In this file, we run the code using files `ncName_ncID_transID.csv`, `eng.csv `and `transcripts_seq.fasta`, which are stored in the `data/human` or  `data/mouse`  folder. The result from running the code are stored in the `result/human` and `result/mouse` folder, named `GIC_score.csv`.

- idmap.py

In this file, we run the code using `LPI.csv`, which are stored in `data/human` or  `data/mouse` folder.  This code generates three files,` id_lncRNA.txt`, `id_protein.txt` and `id_lncRNA_protein.txt` .  The results are stored in` data/human` and  `data/mouse`folder. 

- pro_path.py

In this file, we use the code using `positive.csv` , `negative1.csv` and `negative2.csv`, which  are stored in `result/human`or `result/mouse` folder.  This code also uses text file generated by `metapath2vec.cpp`, e.g., `out_mouse_300_40.txt` are stored in `result/mouse/para/300_40_32`. This code generates two files, `dataset1.csv`and `dataset2.csv` , e.g. these files are stored in  `result/mouse/para/300_40_32`.

- random.py

In this file, this code selects negative samples according to "random" negative sample strategy. For example, In mouse dataset, we use the code using `essential.csv` strored in `data/mouse`folder, and `out_mouse_300_40.txt`stored in `result/mouse/para/300_40_32`. This code generates two files, `mouse_dataset1.csv` and `mouse_dataset2.csv` stored in `data/compare`.
### 1.6 result

- human

In this folder, `positive.csv`  stores 154 human essential lncRNAs(positive samples). `negative1.csv` stores negative samples equal to the number of positive samples(according to "GIC" negative sample strategy ). `negative2.csv`stores negative samples, which are twice as large as positive samples(according to "GIC" negative sample strategy ). `GIC_score.csv` stores the GIC score of all lncRNA genes.  `CNN.txt`,`RF.txt` and `SVM.txt` are stored   the TPR and FPR value of  CNN , RF and SVM model in different datasets.
In `human/para` fold, there are 11 folds. The format of the files : `n_l_r`, n is the number of walks per node, l is the length of each walk , and r is the dimension of embedding representation.  For example, In `300_30_128` folder,  There are three files.  The ratio of positive and negative samples in this `dataset1.csv` is 1:1, and the vectors of all lncRNAs are 128 dimensions, which are generated by the metapath2vec algorithm.  The ratio of positive and negative samples in this `dataset2.csv` is 1:2, and the vectors of all lncRNAs are 128 dimensions, which are generated by the metapath2vec algorithm. `out_human_300_30.txt` are the result of `metapath2vec.cpp` .

- mouse

In this folder, all files have the same distribution as the `result/human` folder, but this folder works for Mouse.  It should be noted that there are 25 essential lncRNAs in mouse dataset.

---

## 2. How to use PredEssLnc
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
./metapath2vec -train ../../data/human/in_human_300_30.txt -output ../../result/human/out_human_300_30 -pp 1 -size 128 -window 7 -negative 5 -threads 32
```
Users can get `out_human_300_30.txt `as output file, then move this file to Windows environment using XFTP software, store it in the `result/human/para/300_30_128` folder.
### 2.3 Select reliable negative samples
Step1:  Users can run `cal_gic.py` to get GIC score of all lncRNAs, input `ncName_ncID_transID.csv`, `eng.csv `and `transcripts_seq.fasta`, output `GIC_score.csv`.
Step2: In `GIC_score.csv` file, the value is sorted in ascending order, and the data with the same and twice the number of positive samples are respectively selected to form `negative1.csv` and `negative2.csv`, which are stored in `result/human` folder. 
Step3:  User can run `pro_path.py`, process the `out_human_300_30.txt` file and select the lncRNA vectors in the `positive.csv`, `negative1.csv `and `negative2.csv` files respectively to form `dataset1.csv`(H1 dataset) and `dataset2.csv`（H2 dataset), which are stored in `result/human/para/300_30_128`folder.
### 2.4 Supervised machine learning
#### 2.4.1 SVM model
| C | gamma | kernel function | dataset |
| --- | --- | --- | --- |
| 1 | 0.1 | rbf | H1 |
| 1 | 0.1 | rbf | H2 |
| 100 | 0.01 | rbf | M1 |
| 10 | 0.01 | rbf | M2 |

The table above shows the parameters used by the SVM model under different datasets.
For example, users can run `svm.py`, input `dataset1.csv`, use leave-one-out cross-validation, and get  accuracy, precision and other 's performance indicators.
#### 2.4.2 RF model
User can run `rf.py` , set max_depth=9, n_estimators=50, random_state=0 in RF model.
#### 2.4.3 CNN model
We constructed a CNN model which contained two convolutional layers. In human , users can run `cnn_human.py`; In mouse, users can run `cnn_mouse.py`.
### 2.5 Other
The `compare.py` is used for comparison experiments with the SGII method. 

