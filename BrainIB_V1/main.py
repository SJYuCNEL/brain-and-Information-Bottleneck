import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform

from utilis import calculate_MI
from data.creat_data import load_data, separate_data
from graphCNNb import GraphCNN
from pre_subgraph import MLP_subgraph
from sklearn.metrics import confusion_matrix


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch, SG_model):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0

    for pos in pbar:

        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]

        batch_subgraph = batch_graph
        for graph in batch_subgraph:
            edge = SG_model(graph)
            graph.edge_mat = edge
        
        embeddings, original_output = model(batch_graph)
        positive, subgraph_output = model(batch_subgraph)

        with torch.no_grad():
            Z_numpy1 = positive.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy1, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma1 = np.mean(np.mean(np.sort(k[:, :10], 1))) 

        with torch.no_grad():
            Z_numpy2 = embeddings.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy2, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma2 = np.mean(np.mean(np.sort(k[:, :10], 1))) 

        mi_loss = calculate_MI(embeddings,positive, sigma2**2,sigma1**2)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        
        #compute loss
        classify_loss = criterion(subgraph_output, labels)
        loss = classify_loss + mi_loss * args.mi_weight
        
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        batch_graph = [graphs[idx] for idx in sampled_idx ]
        if len(sampled_idx) == 0:
            continue
        embedding, score = model(batch_graph)
        output.append(score.detach())
    return torch.cat(output, 0)

def calc_performance_statistics( y_pred, y):

    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    acc = (TN + TP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    prc = TP / (TP + FP)
    f1s = 2 * (prc * sen) / (prc + sen)
    mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

    return acc, sen, spc, prc, f1s, mcc

def test(args, model, device, train_graphs, test_graphs, epoch,SG_model):

    model.eval()

    for graph in train_graphs:
        edge = SG_model(graph)
        graph.edge_mat = edge
    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    for graph in test_graphs:
        edge = SG_model(graph)
        graph.edge_mat = edge
    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    test_loss = criterion(output, labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()
    test_acc, test_sen, test_spc, test_prc, test_f1s, test_mcc = calc_performance_statistics(pred,labels)

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test, test_loss, test_f1s, test_mcc 

# Training settings
# Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
parser.add_argument("--mi_weight", type=float, default= 0.001, help="classifier hidden dims")
parser.add_argument("--weight-decay", type=float, default=0.0001, help="Adam weight decay. Default is 5*10^-5.")
args = parser.parse_args()

#set up seeds and gpu device
torch.manual_seed(0)
np.random.seed(0)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

#load dataset
graph_filename = "data/ABIDE.mat"
graph1 = loadmat(graph_filename)
graphs= load_data(graph1)

##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
num_classes=2
model = GraphCNN(args.num_layers, args.num_mlp_layers, 116, args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

max_acc = 0.0
SG_model = MLP_subgraph(device).to(device)
optimizer = optim.Adam(list(model.parameters()) +
        list(SG_model.parameters()),
        lr = args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(1, args.epochs + 1):
    

    avg_loss = train(args, model, device, train_graphs, optimizer, epoch, SG_model)
    acc_train, acc_test, test_loss, test_f1s, test_mcc = test(args, model, device, train_graphs, test_graphs, epoch,SG_model)
    scheduler.step()
