from Bio import SeqIO
import numpy as np
import torch
import re
import os
base = ["A", "T", "G", "C", "N"]

def to_tensor(string):
    seq = sanitize(string)
    idx = list(map(lambda b: base.index(b), seq))
    idx = np.array(idx)
    seq = np.eye(5, 4)[idx].T
​
    return torch.tensor(seq).unsqueeze(0)
​
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
​
    return seq
​
def read_one(path):
    seq = None
    name = ""
​
    for record in SeqIO.parse(path, "fasta"):
        # only complete genome
        
        seq_ = sanitize(record.seq)
        idx = list(map(lambda b: base.index(b), seq_))
        idx = np.array(idx)
        seq = np.eye(5, 4)[idx]
        name = record.name
​
    return name, seq
​
def read_all(dir):
    species = []
    seqs = []
    labels = np.array([])
​
    for i, filename in enumerate(os.listdir(dir)):

        print("\rLoading... {:0=3}/{:0=3}".format(i+1, len(os.listdir(dir))), end="")
        name, seq = read_one(dir+"/"+filename)
        species.append(name)
        seqs.append(seq)
        labels = np.append(labels, i)
​
    return species, seqs, labels
​
class DataLoader():
    def __init__(self, batch_size, how="random", length=1024, is_train=True, use_all=False, test_size=0.3, n_batch=1000):
        self.is_train = is_train
        self.batch_size = batch_size
        self.how = how
        self.n_batch = n_batch
        self._i = 0
        self.length = length
        self.use_all = use_all
        self.test_size = test_size
​
    def __call__(self, species, seqs, labels):
        self.species, self.seqs, self.labels = species, seqs, labels
        self.lens = np.array([len(x) for x in seqs])
        self.bounds = self.lens if self.use_all else np.array([int(len(x) * self.test_size) for x in self.seqs])
​
    def __iter__(self):
        return self
​
    def __next__(self):
        if self._i >= self.n_batch:
            self._i = 0
            raise StopIteration()
​
        self._i += 1
​
        X_batch, y_batch = self.get_batch()
        X_batch = torch.tensor(X_batch)
        y_batch = torch.tensor(y_batch).long()
​
        return X_batch, y_batch
​
    def __len__(self):
        return self.n_batch
​
    def get_batch(self):
        if self.how=="random":
            idx = np.random.randint(len(self.bounds), size=self.batch_size)
​
            def crop(label, is_train):
                seq = None
                label = label[0]
​
                if is_train:
                    start = np.random.randint(self.bounds[label]-self.length)
                else:
                    start = np.random.randint(self.bounds[label], self.lens[label]-self.length)
​
                return self.seqs[label][start:start+self.length].T
​
            return np.apply_along_axis(crop, 1, idx.reshape(-1, 1), self.is_train), self.labels[idx]
​
    def get_one(self, ind):
        start = np.random.randint(self.lens[ind]-self.length)
        return torch.tensor(self.seqs[ind][start:start+self.length].T).unsqueeze(0)
