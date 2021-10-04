from Bio import SeqIO
import glob
import numpy as np
import torch
import re
import os
base = ["A", "T", "G", "C", "N"]

def to_tensor(seq):
    seq = sanitize(seq)
    idx = list(map(lambda b: base.index(b), seq))
    idx = np.array(idx)
    seq = np.eye(5, 4)[idx].T
    return seq


def sanitize(seq):
    seq = re.sub(r"R", ["A", "G"][np.random.randint(2)], seq)
    seq = re.sub(r"Y", ["T", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"K", ["G", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"S", ["G", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"W", ["A", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"M", ["A", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"B", ["T", "G", "C"][np.random.randint(3)], seq)
    seq = re.sub(r"V", ["A", "G", "C"][np.random.randint(3)], seq)
    seq = re.sub(r"D", ["A", "G", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"H", ["A", "C", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"a", "A", seq)
    seq = re.sub(r"t", "T", seq)
    seq = re.sub(r"g", "G", seq)
    seq = re.sub(r"c", "C", seq)
    return seq

def read_one(path):
    seq = None
    name = ""
    for record in SeqIO.parse(path, "fasta"):
        seq = to_tensor(str(record.seq))
        
        label = str(path)
    return label, seq

def read_all(dir):
    seqs = []
    labels = []
    files = glob.glob(dir+"/*.fna")
    num=0
    for file in files:
        print("\rLoading... {:0=3}/{:0=3}".format(num+1, len(files)), end="")
        maxseqlen = 0
        for i, record in enumerate(SeqIO.parse(file, "fasta")): 
            
            seqlength = len(str(record.seq))
            if maxseqlen < seqlength:
                seq = to_tensor(str(record.seq))
                taxa = record.description
                #g = taxa.split(" ")[1]
                #s = taxa.split(" ")[2]
                maxseqlen = seqlength
        num = num +1
        seqs.append(seq)
        labels.append(taxa)
        #species.append(s)
        #seqlen.append(maxseqlen)
    return seqs, labels
""" 
    for i, record in enumerate(SeqIO.parse(file, "fasta")): 
        maxseqlen=0
        print("\rLoading... {:0=3}/{:0=3}".format(i+1, len(os.listdir(dir))), end="")
        seqlength = len(str(record.seq))
        if maxseqlen < seqlength:
            seq = to_tensor(str(record.seq))
            seqstr = to_tensor(str(record.seq))
            taxa = record.description
            #g = taxa.split(" ")[1]
            #s = taxa.split(" ")[2]
            maxseqlen = seqlength
        num = num +1
        seqs.append(seq)
        seqstrs.append(seq)
        taxa.append(taxa)
    for i, filename in enumerate(os.listdir(dir)):
        
        print("\rLoading... {:0=3}/{:0=3}".format(i+1, len(os.listdir(dir))), end="")
        for record in SeqIO.parse(dir+"/"+filename, "fasta"):
            seq = to_tensor(str(record.seq))
            label = str(filename).split(".")[0]
            seqs.append(seq)
            labels.append(label)
    return seqs, labels
"""
def read_contig(dir):
    labels = []
    seqs = []
    
    for i, filename in enumerate(os.listdir(dir)):
        for record in SeqIO.parse(dir+"/"+filename, "fasta"): 
            print("\rLoading testdata... {:0=3}/{:0=3}".format(i+1, len(os.listdir(dir))), end="")
            seq = to_tensor(str(record.seq))
            seqs.append(seq)
            labels.append(record.description)

    return seqs, labels

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
"""
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
        
    def __call__(self, species, seqs, labels):
        self.species, self.seqs, self.labels = species, seqs, labels
        self.lens = np.array([len(x) for x in seqs])
        self.bounds = self.lens if self.use_all else np.array([int(len(x) * self.test_size) for x in self.seqs])
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i >= self.n_batch:
            self._i = 0
            raise StopIteration()
            
        self._i += 1
        
        X_batch, y_batch = self.get_batch()
        X_batch = torch.tensor(X_batch)
        y_batch = torch.tensor(y_batch).long()
        
        return X_batch, y_batch
    
    def __len__(self):
        return self.n_batch
    
    def get_batch(self):
        if self.how=="random":
            idx = np.random.randint(len(self.bounds), size=self.batch_size)
            
            def crop(label, is_train):
                seq = None
                label = label[0]
                
                if is_train:
                    start = np.random.randint(self.bounds[label]-self.length)
                else:
                    start = np.random.randint(self.bounds[label], self.lens[label]-self.length)
                    
                return self.seqs[label][start:start+self.length].T
            
            return np.apply_along_axis(crop, 1, idx.reshape(-1, 1), self.is_train), self.labels[idx]
        
    def get_one(self, ind):
        start = np.random.randint(self.lens[ind]-self.length)
        return torch.tensor(self.seqs[ind][start:start+self.length].T).unsqueeze(0)
"""
