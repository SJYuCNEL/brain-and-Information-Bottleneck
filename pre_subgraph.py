import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

# Subgraph Generator
class MLP_subgraph(nn.Module):
    def __init__(self,device):
        super(MLP_subgraph, self).__init__()
        self.num_nodes = 116
        self.hidden_dim =32
        self.linear1 = nn.Linear(self.num_nodes,self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_nodes)
        self.device = device


    def forward(self, graph):
        
        node_features = graph.node_features.to(self.device)
        Adj = graph.adj.to(self.device) # graph adjacency matrix 

        node_feature_1 = F.relu(self.linear1(node_features)) 
        node_feature_2 =self.linear2(node_feature_1)

        node_mask = torch.sigmoid(node_feature_2)
        node_group = node_mask.view(int(116*116/2),2) # reshape edge attention mask

        self.drop_mask = F.gumbel_softmax(node_group , tau=1, hard=True) # Gumbel_softmax
        self.drop_mask_hard = self.drop_mask.view(116,116) #generate edge assignment
        
        self.drop_mask_hard = self.drop_mask_hard * Adj 
        edge = dense_to_sparse(self.drop_mask_hard)[0]

        return edge
