import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import dense_to_sparse
import math

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.edge_index = []

        self.max_neighbor = 0


def load_data(graph):

    print('loading data')
    num_graphs=graph["label"].size
    label1=graph["label"]
    label=np.append(label1,label1)
    g=nx.Graph()
    g_list=[]
    n=116 # num of nodes
    for i in range(num_graphs):  
        node_features = torch.FloatTensor(graph["graph_struct"][0][i][1])
        tepk = node_features.reshape(-1,1)
        tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(math.pow(node_features.shape[0],2) / 20*2)]
        adj = torch.Tensor(np.where(node_features > mk, 1, 0))
        edge_index=dense_to_sparse(adj)[0]
        node_tags=list(range(0, 116))
        for j in range(n):
            g.add_node(j)
            edge_neighbor=torch.where(adj[j])[0]
            row=len(edge_neighbor)
            l=label[i]
            for k in range(row-1):
                g.add_edge(j, edge_neighbor[k].item())
        g_list.append(S2VGraph(g, l, node_tags))
        g_list[i].node_features = node_features
        g_list[i].edge_index = edge_index
        g_list[i].adj = adj
        g_list[i].edge_mat = edge_index

    print("# data: %d" % len(g_list))

    return g_list

def separate_data(graph_list,  seed, fold_idx):
    
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]

    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    return torch.exp(-dist /sigma)

def reyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

# calculate mutual information
def calculate_MI(x,y,s_x,s_y):
    
    Hx = reyi_entropy(x,sigma=s_x)
    Hy = reyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    Ixy = Hx+Hy-Hxy
    
    return Ixy