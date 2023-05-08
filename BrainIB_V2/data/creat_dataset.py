import os
import scipy.io as scio
from scipy.sparse import coo_matrix

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def read_dataset():
    PATH = os.getcwd()
    data = scio.loadmat(PATH+'/data/md_AAL_0.4.mat')
    # print(len(data['label']))
    dataset = []
    for graph_index in range(len(data['label'])):
        label = data['label']
        # print(data.keys())
        # print(len(data['graph_struct'][0]))
        # print(len(data['label']))

        # edge为该图的权重
        graph_struct = data['graph_struct'][0]


        edge = torch.Tensor(graph_struct[graph_index]['edge'])
        
        # print(edge[:, :2].t().shape)
        # print(edge[:, -1].reshape(-1, 1).shape)

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])
        # print(ROI.shape)
        node_tags = torch.Tensor(graph_struct[graph_index]['node_tags'])
        adj = torch.Tensor(graph_struct[graph_index]['adj'])
        neighbor = graph_struct[graph_index]['neighbor']
        y = torch.Tensor(label[graph_index])
        A = torch.sparse_coo_tensor(
            indices = edge[:, :2].t().long(),
            values = edge[:, -1].reshape(-1,).float(),
            size = (116, 116)
            )
        G = (A.t() + A).coalesce()

        # print(edge.indices().shape)
        # print(G.indices().shape)

        graph = Data(x=ROI.reshape(-1,116).float(),
                     edge_index=G.indices().reshape(2,-1).long(),
                     edge_attr=G.values().reshape(-1,1).float(),
                     y=y.long())
        dataset.append(graph)
        # print(adj)
        # print(graph.edge_index.shape)

    # print(dataset)
    
    # with open(PATH + '/data/asd_dataset.txt', 'w+') as f:
    #     f.writelines(dataset)
    return dataset

if __name__ == '__main__':
    dataset = read_dataset()
    print(len(dataset))
    loader = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print(loader.y)