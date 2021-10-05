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

Regarding model learning, you have two options:  
1.use the trained model  
2.train the model

First, the points common to both patterns are shown below.
* --contig `<CONTIG>`  
Where contigs are located. Sequence data must be in fasta format.  
default:  
* --layer `<LAYER>`  
Style matrices are calculated in `<LAYER>` of the model.  
default:4  
* --dir `<DIRECTORY>`  
All training data should be in `<DIRECTORY>`  
default:
* --

1.use the trained model


2.train the model
All training data should be in `<DIRECTORY>`, and Style matrices are calculated in `<LAYER>` of the model. `FILEPATH` is where contigs are located. Sequence data must be in fasta format. 


```
python main.py --layer <LAYER> --dir <DIRECTORY> --contig <FILEPATH>
```

If you try other training parameters, just add `--rate` and `--epoch` arguments. And also you can control loggin level with `--verbose` argument.

```
python main.py --layer 4 --dir ./data --contig test.fasta --rate 0.001 --epoch 100 --verbose 2
```
