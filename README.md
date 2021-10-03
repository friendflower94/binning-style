# Genomic style: a deep-learning approach to characterize bacterial genome sequences
## Overview
You can extract features of bacterial genome sequences with a deep learning approach.

## Requirements
* python 3.6.5 (with following packages)
  * biopython 1.79
  * torch 1.9.1
  * numpy 1.16.4
  * bio 1.1.3

## Usage
You can train a deep-learning model and extract style matrices running a following command.
All training data should be in `<DIRECTORY>`, and Style matrices are calculated in `<LAYER>` of the model. `FILEPATH` is where contigs are located. Sequence data must be in fasta format. 

```
python main.py --layer <LAYER> --dir <DIRECTORY> --contig <FILEPATH>
```

If you try other training parameters, just add `--rate` and `--epoch` arguments. And also you can control loggin level with `--verbose` argument.

ex)
```
python main.py --layer 4 --dir ./data --contig test.fasta --rate 0.001 --epoch 100 --verbose 2
```
