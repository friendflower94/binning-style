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
  
* --layer `<LAYER>`  
Style matrices are calculated in `<LAYER>` of the model.  
default:4  
* --dir `<DIRECTORY_train>`  
All training data should be in `<DIRECTORY_train>`.  
Required when training a model.  
Please use the file name as the bacterial species name or taxonomy.  
ex)  
default: ./data_139  
* --contig, -c `<DIRECTORY_contig>`  
All DNA sequences you want to bin should be in `<DIRECTORY_contig>`.  
Sequence data must be in fasta format.  
default: ./testdata
* --bin, -b `<BIN>`  
Split the input DNA sequences in `<DIRECTORY_contig>` into `<BIN>` bins.  
default:60 
* --verbose, -v
You can specify whether to train the model.
Select 0 if you want to use the trained model and 2 if you want to train a model.  
default: 0  
* --outofbin `<FILENAME>`
Output binning result to <FILENAME>.  
default: ./binningresult.txt  

Regarding model learning, you have two options:  
1.use the trained model  
```
python main.py --layer <LAYER> --contig <DIRECTORY> --verbose 0
```

2.train the model
```
python main.py --layer <LAYER> --dir <DIRECTORY> --contig <DIRECTORY> --verbose 2
```

2.train the model
All training data should be in `<DIRECTORY>`, and Style matrices are calculated in `<LAYER>` of the model. `FILEPATH` is where contigs are located. Sequence data must be in fasta format. 


```
python main.py --layer <LAYER> --dir <DIRECTORY> --contig <FILEPATH>
```

If you try other training parameters, just add `--rate` and `--epoch` arguments. And also you can control loggin level with `--verbose` argument.

```
python main.py --layer 4 --dir ./data --contig test.fasta --rate 0.001 --epoch 100 --verbose 2
```
