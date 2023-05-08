from tqdm import tqdm
import copy

from torch_geometric.nn import MessagePassing

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric
from torch_geometric.loader import DataLoader

import numpy as np

from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist, squareform

import time


def separate_data(graph_list, seed, fold_idx):
    """
    Separate the dataset into trainsets and testsets (list of graph)
    """
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [graph.y.numpy() for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list


def pairwise_distances(x):
    # x should be two dimensional
    # The latter dimension of x is the number of features of the graph embedding after GNN
    x = x.view(-1, 1024)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)


def renyi_entropy(x,sigma):
    """
    Function for computing matrix entropy.
    """
    alpha = 5
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    """
    Function for computing joint matrix entropy.
    """
    alpha = 5
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x,y,s_x,s_y):
    """
    Function for computing mutual information using matrix entropy
    """
    Hx = renyi_entropy(x,sigma=s_x)
    Hy = renyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    Ixy = Hx+Hy-Hxy
    #normlize = Ixy/(torch.max(Hx,Hy))
    
    return Ixy


def train(args, model, train_dataset, optimizer, epoch, SG_model, device, criterion = nn.CrossEntropyLoss()):
    """
    A function used to train the model that feeds all the training data into the model once per execution
    """
    # model.train()
    # SG_model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    miloss_accum = 0
    # train_subgraph = copy.deepcopy(train_dataset)
    total_time = 0
    for pos in pbar:
        indices = range(0, len(train_dataset), args.batch_size)
        
        for i in indices:
            model.train()
            SG_model.train()
            graphs = train_dataset[i : i + args.batch_size]
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))

            embeddings, original_output = model(batch_graph)
            
            subgraphs = copy.deepcopy(graphs)

            positive_penalty = torch.Tensor([0.0]).float().to(device)
           
            for graph in subgraphs:
                subgraph, pos = SG_model(graph)
                graph = subgraph.to(device)
                positive_penalty += pos

            positive_penalty = (positive_penalty / len(subgraphs))
            batch_subgraph = next(iter(DataLoader(subgraphs, batch_size=len(subgraphs))))

            positive, subgraph_output = model(batch_subgraph)
            
            # calculate to sigma1 and sigma2
            with torch.no_grad():
                Z_numpy1 = embeddings.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy1, 'euclidean'))
                k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
                sigma1 = np.mean(np.sort(k[:, :10], 1))

            with torch.no_grad():
                Z_numpy2 = positive.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy2, 'euclidean'))
                k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
                sigma2 = np.mean(np.sort(k[:, :10], 1))


            mi_loss = calculate_MI(embeddings, positive, sigma1**2, sigma2**2) / len(graphs)
            labels = batch_graph.y.view(-1,).to(device)

            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            classify_loss = criterion(subgraph_output, labels)
            loss = classify_loss + mi_loss * args.mi_weight



            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.detach().cpu().numpy()

            loss_accum += loss
            miloss_accum += mi_loss
            
            pbar.set_description(f'epoch: {epoch}')

    print(loss_accum)
    print(len(indices))
    average_loss = loss_accum / len(indices)
    average_miloss = miloss_accum / len(indices)
    print(f"Loss Training: {average_loss}\tMutual Information Loss: {average_miloss}")
    return average_loss, mi_loss


def test(args, model, train_dataset, test_dataset, SG_model, device, criterion=nn.CrossEntropyLoss()):
    """
    A function used to test the trained model that feeds all the testing data into the model once per execution
    """
    model.eval()
    SG_model.eval()

    train_dataset = copy.deepcopy(train_dataset)
    test_dataset = copy.deepcopy(test_dataset)

    num_of_edges_pre = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))).edge_index.shape[1]
    for graph in train_dataset:
        subgraph, pos = SG_model(graph)
        graph = subgraph
    num_of_edges_after = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))).edge_index.shape[1]

    print(f'edge number (pre/after): ({num_of_edges_pre}/{num_of_edges_after})')
    

    total_correct_train = 0
    for train_dataset_batch in iter(DataLoader(train_dataset, batch_size=args.batch_size)):
        _, output_train = model(train_dataset_batch)
        _, y_hat_train = torch.max(output_train, dim=1)
        labels_train = train_dataset_batch.y.view(-1).to(device)

        correct = torch.sum(y_hat_train == labels_train)
        total_correct_train += correct
    
    acc_train = total_correct_train / float(len(train_dataset))
    print(f'train (correct/samples) : ({total_correct_train}/{len(train_dataset)})')

    for graph in test_dataset:
        subgraph, pos = SG_model(graph)
        graph = subgraph

    total_correct_test = 0
    for test_dataset_batch in iter(DataLoader(test_dataset, batch_size=args.batch_size)):
        _, output_test = model(test_dataset_batch)
        _, y_hat_test = torch.max(output_test, dim=1)
        labels_test = test_dataset_batch.y.view(-1,).to(device)
        test_loss = criterion(output_test, labels_test)
        correct = torch.sum(y_hat_test == labels_test)
        total_correct_test += correct

    acc_test = total_correct_test / float(len(test_dataset))
    print(f'test (correct/samples): ({total_correct_test}/{len(test_dataset)})')

    print("accuracy (train/test): (%f/%f)" % (acc_train, acc_test))

    return acc_train, acc_test, test_loss