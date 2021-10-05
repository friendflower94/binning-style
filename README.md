# Genomic style: a deep-learning approach to characterize bacterial genome sequences
## Overview
Binning is the process of clustering DNA sequence into bins of the same species from a DNA sequences of a sample in which multiple bacterial species are mixed.  
You can divide the DNA sequences into bins according to the genomic features of bacterial genome sequences, "genomic style", with a deep learning approach, "genomic style".

## Requirements
* python 3.7.3 (with following packages)
  * torch 1.9.1
  * numpy 1.16.4
  * bio 1.1.3
  * biopython 1.79

## Usage
Parameter and command examples are shown below.  

* --layer, -l `<LAYER>`  
Style matrices are calculated in `<LAYER>` of the model.  
1-6 can be selected in the current model.  
default:4  

* --dir, -d `<DIRECTORY_train>`  
All training data should be in `<DIRECTORY_train>`.  
Required when training a model.  
Read the longest DNA sequence in each file.  
Please use the file name as the bacterial species name or taxonomy.  
ex)bacteria__actinobacteria__actinobacteria__corynebacteriales__corynebacteriaceae__corynebacterium__diphtheriae.fasta
default: ./data_139  

* --contig, -c `<DIRECTORY_contig>`  
All DNA sequences you want to bin should be in `<DIRECTORY_contig>`.  
Sequence data must be in fasta format.  
default: ./test  

* --bin, -b `<BIN>`  
Split the input DNA sequences in `<DIRECTORY_contig>` into `<BIN>` bins.  
default:60 

* --train, -t  
You can specify whether to train the model.  
Select 0 if you want to use the trained model and 2 if you want to train a model.  
default: 0  

* --model, -m `<MODEL>`  
Load the model of `<MODEL>`.  
default: ./weight/modelweight.weight 

* --out, -o `<FILENAME>`  
Output binning result to <FILENAME>.  
default: ./binningresult.txt  

Regarding model learning, you have two options:  

**1.use the trained model **
```
python main.py --layer <LAYER> --contig <DIRECTORY_contig> --train 0 --model <MODEL>
```  
ex)
```
python main.py --layer 4 --contig ./test --train 0 --model ./weight/modelweight.weight
```

**2.train the model**
```
python main.py --layer <LAYER> --dir <DIRECTORY_train> --contig <DIRECTORY_contig> --train 2
```
ex)
```
python main.py --layer 4 --dir ./trainingdata_139 --contig ./test --train 2
```

If you try other training parameters, add `--rate` and `--epoch` arguments.

* --rate, -r  
learning rate  
default: 0.001  

* --epoch, -e  
epoch for training a model    
default: 0.001  

ex)
```
python main.py --layer 4 --dir ./trainingdata_139 --contig ./test --train 2 --rate 0.01 --epoch 200
```
