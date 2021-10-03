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
    parser.add_argument("-o", "--out", help="directory that outputs the binning result", default="./out")
    parser.add_argument("-v", "--verbose", help="2 when training the model, 0 when using the weights provided", type=int, default=0)
    parser.add_argument("--weight", "-w", help="model's weight file", default="--model=./weight/modelweight.weight")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # read trainingdata
    if args.verbose > 1:
        print("Reading training data...")
        species, seqs, labels = read_all(args.dir)
        train_loader = DataLoader(length=1024,batch_size=128,n_batches=1000)
        test_loader = DataLoader(length=1024,batch_size=128, n_batches=1000)
        train_loader(species, seqs, labels)
        test_loader(species, seqs, labels)
    
        # train model
        if args.verbose > 1: print("\nTraining model...")
    
        model = Discriminator(1024, len(species)).double().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.rate)

        for epoch in range(args.epoch):
            train(model, device, loader, optimizer, epoch+1)
            val_loss = test(model, device, test_loader)
            print("")
    
    if args.verbose < 1:
        model = Discriminator(1024, 139).float().to(device)
        model.load_state_dict(torch.load(args.weight))
        x = torch.ones(1, 4,1024).double().to(device)
        x = Variable(x, requires_grad=True)
    
    
    # read testdata
    species_test, seqs_test, labels_test = read_all(args.contig)
    
    ## calculate style matrix
    def stylematrix(seq):
        style = model.cuda().get_style(seq,args.layer)
        style = style.cpu().detach().squeeze().numpy()
        style = style[np.triu_indices(style.shape[0])]
        return style
    
    ## calculate style
    def calculate_style(seq):
        length = seq.shape[-1]
        
        if length > 1024:
            split_num = (length-1024)//500 +1
            for i in range (split_num):
                style_split = stylematrix(seq[:,:,:1024])
                if i != 0:
                    style = np.vstack([style,style_split])
                else:
                    style = style_split
                seq = seq[:,:,500:]
            style = np.average(style, axis=0)
            
        else:
            seq = F.pad(seq,(0,1024-length))
            style = stylematrix(seq)
        return style
    
    # calculate style matrices    
    styles = []
    for i in range(len(seqs)):
        print("\rCalculating... {:0=3}".format(i+1), end="")
        style = calculate_style(seqs[i])
        styles.append(style)
        
    # label encode
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le = le.fit(labels_test)
    truelabel = le.transform(labels_test)
    
    # Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    result = AgglomerativeClustering(affinity='euclidean',
                                     linkage='ward',
                                     n_clusters=np.unique(truelabel).shape,
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
