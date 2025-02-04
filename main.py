import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from model import Discriminator
from loader import DataLoader,read_all, read_contig, to_tensor
from Bio import SeqIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", help="layer for calculating style matrices", type=int, default=4)
    parser.add_argument("-e", "--epoch", help="epoch for training a model", type=int, default=100)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("-d", "--dir", help="directory that contains fasta files for training", default="./trainingdata_139")
    parser.add_argument("-c", "--contig", help="directory that contains fasta files of contigs", default="./test")
    parser.add_argument("-o", "--out", help="directory that outputs the binning result", default="./binnigresult.txt")
    parser.add_argument("-t", "--train", help="2 when training the model, 0 when using the saved model", type=int, default=0)
    parser.add_argument("-m", "--model", help="path of the saved model", default="./weight/modelweight.weight")
    parser.add_argument("-n", "--numofbin", help="num of bins", default=60)
    args = parser.parse_args()
    
    def train(model, device, loader, optimizer, epoch):
        model.train()
        data_size = len(loader)
        print(data_size)
        start = time.time()
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            y = torch.from_numpy(y).clone()
            y = y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            print("\rTrain Epoch: {} [ {:0=5}/{:0=5} ({:.0f}%)]\t Loss: {:.4f}".format(epoch,
            batch_idx+1, data_size, (batch_idx + 1) * 100. / data_size, loss.item()), end="")
        end = time.time()
        print("\tThis took {:.3f} seconds".format(end-start), end="")
        
    def test(model, device, loader):
        model.eval()
        val_loss = 0
        true = 0
        
        data_size = len(loader) * loader.batch_size
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += F.nll_loss(out, y, reduction="sum").item()
                y_pred = out.argmax(dim=1, keepdim=True)
                true += y_pred.eq(y.view_as(y_pred)).sum().item()
        val_loss /= data_size
        true /= data_size
        val_losses.append(val_loss)
        accuracy.append(true)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%".format(val_loss, true*100))
        return val_loss
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # calculate style matrix 
    def stylematrix(seq):
        style = model.cuda().get_style(seq,args.layer)
        style = style.cpu().detach().squeeze().numpy()
        style = style[np.triu_indices(style.shape[0])]
        return style
    
    # calculate style
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
 
    # 1-1)trainig model
    if args.train > 1:
        # read trainingdata
        print("Reading training data...")
        seqs, labels = read_all(args.dir)
        for i in range (len(seqs)):
            seqs[i] = torch.tensor(seqs[i]).unsqueeze(0).float()
        le = LabelEncoder()
        le = le.fit(labels)
        labels_en = le.transform(labels)
        print("-->Complete reading training data")
        print("-->num of training data:", len(labels))
        print(labels[0])
        print(seqs[0])
        print(seqs[0].shape)
        train_loader = DataLoader(length=1024,batch_size=64,n_batches=1000)
        train_loader(labels_en, seqs, labels_en)
        print(len(train_loader))

        # train model
        print("\nTraining model...")
        model = Discriminator(1024, len(labels)).float().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.rate)

        for epoch in range(args.epoch):
            train(model, device, train_loader, optimizer, epoch+1)
            #val_loss = test(model, device, test_loader)
            print("")
            
    # 1-2)using trained model
    if args.train < 1:
        print("Loading trained model:", str(args.model))
        model = Discriminator(1024, 139).float().to(device)
        if device =="cuda":
            model.load_state_dict(torch.load(args.model))
        else:
            model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        print("-->Completed loading trained model")
    
    # 2)read testdata
    seqs_test, labels_test = read_contig(args.contig)
    print("\n-->Completed loading testdata")
    print("-->num of contig:", len(labels_test))
    
    # 3)calculate style matrices    
    styles = []
    for i in range(len(seqs_test)):
        print("\rCalculating style matrix... {:0=3}/{:0=3}".format(i+1, len(seqs_test)), end="")
        style = calculate_style(torch.tensor(seqs_test[i]).unsqueeze(0).float().to(device))
        styles.append(style)
    print("\n-->Completed calculating style matrix")
    
    # 4)Agglomerative Clustering
    print("Clustering...")
    from sklearn.cluster import AgglomerativeClustering
    result = AgglomerativeClustering(affinity='euclidean',
                                     linkage='ward',
                                     n_clusters=args.numofbin,
                                     distance_threshold=None).fit(styles)
    predictlabel = result.labels_
    print("-->Completed clustering")
    print("-->num of bins generated:", args.numofbin)
    
    
    # 5)output result
    print("Outputting...")
    with open(args.out, mode='w') as f:
        for i in range (len(set(predictlabel))):
            f.write(">bin_"+str(i)+"\n")
            index=[j for j, x in enumerate(predictlabel) if x == i]
            for k in index:
                f.write(labels_test[k]+"\n")
    print("-->Completed outputting binning result:", str(args.out))
    
    ### label encode
    #from sklearn.preprocessing import LabelEncoder
    #le = LabelEncoder()
    #le = le.fit(labels_test) 
    #truelabel = le.transform(labels_test)
    
    ### evaluate clustering accuracy
    #from sklearn.metrics.cluster import adjusted_rand_score,homogeneity_score,completeness_score
    #ari = adjusted_rand_score(truelabel,predictlabel)
    #print("ARI = {:.3f}" .format(ari))
    #homogeneity = homogeneity_score(truelabel,predictlabel)
    #print("homegeneity = {:.3f}" .format(homogeneity))
    #completeness = completeness_score(truelabel,predictlabel)
    #print("completeness = {:.3f}" .format(completeness))
