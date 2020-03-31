import argparse
import torch
import torch.optim as optim
import numpy as np
from model import Discriminator, train, test
from loader import DataLoader, read_all, to_tensor
from Bio import SeqIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", help="layer for calculating style matrices", type=int, default=4)
    parser.add_argument("-e", "--epoch", help="epoch for training a model", type=int, default=100)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("-d", "--dir", help="directory that contains fasta files for training", default="./trainingdata_139")
    parser.add_argument("-c", "--contig", help="directory that contains fasta files of contigs", default="./test")
    parser.add_argument("--verbose", help="logging level", type=int, default=2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # read trainingdata
    if args.verbose > 1: print("Reading fasta...")
    species, seqs, labels = read_all(args.dir)

    loader = DataLoader(batch_size=128, use_all=True, n_batch=1000)
    loader(species, seqs, labels)
    

    # train model
    if args.verbose > 1: print("\nTraining model...")
    
    model = Discriminator(1024, len(species)).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.rate)

    for epoch in range(args.epoch):
        train(model, device, loader, optimizer, epoch+1)
        print("")

    # calculate style matrices
    if args.verbose > 1: print("Extracting style matrices...")
    
    for record in SeqIO.parse(args.contig, "fasta"):
        tensor = to_tensor(str(record.seq))
        style_matrix = model.get_style(tensor.to(device), args.layer)
       
  
    # Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    result = AgglomerativeClustering(affinity='euclidean',
                                     linkage='ward',
                                     n_clusters=92,
                                     distance_threshold=None).fit(styles)
    predictlabel = result.labels_
        
    # evaluate clustering accuracy

    
    from sklearn.metrics.cluster import adjusted_rand_score,homogeneity_score,completeness_score
    ari = adjusted_rand_score(truelabel,predictlabel)
    print("ARI = {:.3f}" .format(ari))
    homogeneity = homogeneity_score(truelabel,predictlabel)
    print("homegeneity = {:.3f}" .format(homogeneity))
    completeness = completeness_score(truelabel,predictlabel)
    print("completeness = {:.3f}" .format(completeness))
