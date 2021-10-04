import argparse
import torch
import torch.optim as optim
import numpy as np
from model import Discriminator
from loader import DataLoader, read_all, read_contig, to_tensor
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
    parser.add_argument("-m", "--model", help="path of saved model", default="./weight/modelweight.weight")
    
    args = parser.parse_args()
    
    def train(model, device, loader, optimizer, epoch):
        model.train()
        data_size = len(loader)
        
        start = time.time()
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            
            print("\rTrain Epoch: {} [ {:0=5}/{:0=5} ({:.0f}%)]\t Loss: {:.4f}".format(epoch,batch_idx+1, data_size, (batch_idx + 1) * 100. / data_size, loss.item()), end="")
            
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
    
    
    # read trainingdata
    if args.verbose > 1:
        print("Reading training data...")
        species, seqs, labels = read_all(args.dir)
        train_loader = DataLoader(length=1024,batch_size=128,n_batches=1000)
        train_loader(species, seqs, labels)
    
        # train model
        if args.verbose > 1: print("\nTraining model...")
    
        model = Discriminator(1024, len(species)).double().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.rate)

        for epoch in range(args.epoch):
            train(model, device, train_loader, optimizer, epoch+1)
            #val_loss = test(model, device, test_loader)
            print("")
    
    if args.verbose < 1:
        model = Discriminator(1024, 139).float().to(device)
        if device =="cuda":
            model.load_state_dict(torch.load(args.model))
        else:
            model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        #x = torch.ones(1, 4,1024).double().to(device)
        #x = Variable(x, requires_grad=True)
    
    
    # read testdata
    seqs_test, labels_test = read_contig(args.contig)
    
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
    for i in range(len(seqs_test)):
        print("\rCalculating... {:0=3}".format(i+1), end="")
        style = calculate_style(torch.tensor(seqs_test).unsqueeze(0).float().to(device))
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
