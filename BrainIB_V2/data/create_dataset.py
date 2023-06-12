import os
import scipy.io as scio
from scipy.sparse import coo_matrix

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def read_dataset():
    """
    read data from PATH+'/data/md_AAL_0.4.mat' and reconstruct data as 'torch_geometric.data.Data'
    """
    PATH = os.getcwd()
    data = scio.loadmat(PATH+'/data/md_AAL_0.4.mat')   # Data is available at google drive (https://drive.google.com/drive/folders/1EkvBOoXF0MB2Kva9l4GQbuWX25Yp81a8?usp=sharing).
    dataset = []
    for graph_index in range(len(data['label'])):
        label = data['label']

        graph_struct = data['graph_struct'][0]

        edge = torch.Tensor(graph_struct[graph_index]['edge'])

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])

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

        graph = Data(x=ROI.reshape(-1,116).float(),
                     edge_index=G.indices().reshape(2,-1).long(),
                     edge_attr=G.values().reshape(-1,1).float(),
                     y=y.long())
        dataset.append(graph)
    return dataset

if __name__ == '__main__':
    dataset = read_dataset()
    print(len(dataset))
    loader = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print(loader.y)
