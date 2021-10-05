# Genomic style: a deep-learning approach to characterize bacterial genome sequences
## Overview
Binning is the process of clustering DNA sequences into bins of the same species in a mixed sample of DNA sequences from multiple species
You can divide the DNA sequences into bins according to the genomic features of bacterial genome sequences with a deep learning approach, "genomic style".

## Requirements
* python 3.6.5 (with following packages)
  * biopython 1.79
  * torch 1.9.1
  * numpy 1.16.4
  * bio 1.1.3

## Usage



First, the points common to both patterns are shown below.
  
* --layer, -l `<LAYER>`  
Style matrices are calculated in `<LAYER>` of the model.  
default:4  
* --dir, -d `<DIRECTORY_train>`  
All training data should be in `<DIRECTORY_train>`.  
Required when training a model.  
Please use the file name as the bacterial species name or taxonomy.  
ex)  
default: ./data_139  
* --contig, -c `<DIRECTORY_contig>`  
All DNA sequences you want to bin should be in `<DIRECTORY_contig>`.  
Sequence data must be in fasta format.  
default: ./test
* --bin, -b `<BIN>`  
Split the input DNA sequences in `<DIRECTORY_contig>` into `<BIN>` bins.  
default:60 
* --verbose, -v
You can specify whether to train the model.
Select 0 if you want to use the trained model and 2 if you want to train a model.  
default: 0  
* --model, -m `<MODEL>`
Load the model of `<MODEL>`.
default: ./weight/modelweight.weight 
* --numofbin, -n `<FILENAME>`
Output binning result to <FILENAME>.  
default: ./binningresult.txt  

Regarding model learning, you have two options:  
1.use the trained model  
```
python main.py --layer <LAYER> --contig <DIRECTORY> --verbose 0 --model <MODEL>
```  
ex)
```
python main.py --layer 4 --contig ./test --verbose 0 --model ./weight/modelweight.weight
```

2.train the model
```
python main.py --layer <LAYER> --dir <DIRECTORY> --contig <DIRECTORY> --verbose 2
```
ex)
```
python main.py --layer 4 --dir ./trainingdata_139 --contig ./test --verbose 2
```

If you try other training parameters, just add `--rate` and `--epoch` arguments.  
And also you can control loggin level with `--verbose` argument.
* --rate, -r  
learning rate  
default: 0.001  
* --epoch, -e  
epoch for training a model    
default: 0.001  

ex)
```
python main.py --layer 4 --dir ./trainingdata_139 --contig ./test --rate 0.01 --epoch 200 --verbose 2
```
