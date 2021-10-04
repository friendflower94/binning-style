from Bio import SeqIO
import os
import numpy as np
import torch
import re

##### Loading Data #####
base = ["A", "T", "G", "C", "N"]

### function sanitize replaces ambiguous symbols in a sequence with "A", "T", "G" and "C".
## Arguments
# seq: genome sequence in String
## Return Values
# seq: genome sequence in String without ambiguous symbols
def sanitize(seq):
    seq = re.sub(r"R", ["A", "G"][np.random.randint(2)], seq)
    seq = re.sub(r"Y", ["T", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"K", ["G", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"S", ["G", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"W", ["A", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"B", ["T", "G", "C"][np.random.randint(3)], seq)
    seq = re.sub(r"D", ["A", "G", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"H", ["A", "C", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"M", ["A", "C"][np.random.randint(2)], seq)

    return seq

### function to_tensor transforms genome sequences to one-hot vectors
## Arguments
# seq: genome sequence in String
## Return Values
# One-Hot expression of genome sequence in torch.Tensor with shape (1, 4, LENGTH)
def to_tensor(seq):
    seq = sanitize(seq)
    idx = list(map(lambda b: base.index(b), seq))
    idx = np.array(idx)
    seq = np.eye(5, 4)[idx].T

    return torch.tensor(seq).unsqueeze(0).float()

### function read_fasta reads a fasta file and convert it into String
## Arguments
# path: path of fasta file
## Return Values
# seq: genome sequence in torch.Tensor
def read_fasta(path):
    seq = None

    for record in SeqIO.parse(path, "fasta"):
        # only complete genome
        if record.description.find("complete genome") >= 0:
            seq = to_tensor(str(record.seq))

            break

    return seq

### function read_all executes read_fasta on all fasta files
## Arguments
# dir: path of directory
## Return Values
# species: name of species in dataset (list of String)
# seqs: one-hot tensors of genome sequences (list of torch.Tensor)
# labels: filenames of fasta files (used for hierarchical labeling) (array of shape (K, 6))
def read_all(dir):
    species = []
    seqs = []
    labels = np.empty((0, 6))
    files = os.listdir(dir)

    for i, filename in enumerate(files):
        taxa = filename.split(".")[0].split("__")
        name = taxa[-1]

        print("\rLoading... {:0=3}/{:0=3}".format(i+1, len(files)), end="")

        seq = read_fasta(dir+"/"+filename)

        if seq is None: continue
        
        species.append(name)
        seqs.append(seq)
        labels = np.append(labels, np.array(taxa)[np.newaxis, 1:], axis=0)

    return species, seqs, labels

class DataLoader():
    def __init__(self, length, n_batches, batch_size=64, how="random"):
        self.batch_size = batch_size
        self.how = how
        self.n_batches = n_batches
        self._i = 0
        self.length = length
        
    def __call__(self, species, seqs, labels):
        self.species, self.seqs, self.labels = species, seqs, labels
        self.lens = np.array([seq.shape[2] for seq in seqs])
        
    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self.n_batches:
            self._i = 0
            raise StopIteration()

        self._i += 1

        X_batch, y_batch = self.get_batch()

        return X_batch, y_batch

    def __len__(self):
        return self.n_batches

    def get_batch(self):
        if self.how=="random":
            # select species
            X_batch = []
            indexes = np.random.randint(len(self.species), size=self.batch_size)
            lens = self.lens[indexes]

            for index, length in zip(indexes, lens):
                start = np.random.randint(length - self.length)
                X_batch.append(self.seqs[index][:,:,start:start+self.length])

            return torch.cat(X_batch), self.labels[indexes]

    def train_test_split(self, test_size=0.2):
        train_loader = DataLoader(self.length, self.n_batches, self.batch_size, self.how)
        test_loader = DataLoader(self.length, 100, self.batch_size, self.how)

        bounds = np.array([int(x * (1-test_size)) for x in self.lens])
        train_seq = [seq[:,:,:bound] for seq, bound in zip(self.seqs, bounds)]
        test_seq = [seq[:,:,bound:] for seq, bound in zip(self.seqs, bounds)]

        train_loader(self.species, train_seq, self.labels)
        test_loader(self.species, test_seq, self.labels)

        return train_loader, test_loader
    
    def sample(self, size, index):
        print("Sampling {}".format(self.species[index]))
        
        X_batch = []
        indexes = (np.zeros(size) + index).astype(np.int64)
        lens = self.lens[indexes]

        for index, length in zip(indexes, lens):
            start = np.random.randint(length - self.length)
            X_batch.append(self.seqs[index][:,:,start:start+self.length])

        return torch.cat(X_batch)