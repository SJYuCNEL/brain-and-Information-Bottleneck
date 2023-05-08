import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing

class MLP_subgraph(nn.Module):
    def __init__(self,node_features_num, edge_features_num, device):
        super(MLP_subgraph, self).__init__()
        self.device = device
        self.node_features_num = node_features_num
        self.edge_features_num = edge_features_num
        self.mseloss = torch.nn.MSELoss()
        self.feature_size = 64
        self.linear = nn.Linear(self.node_features_num, self.feature_size).to(self.device)
        self.linear1 = nn.Linear(2 * self.feature_size, 8).to(self.device)
        # self.linear1 = nn.Linear(self.node_features_num * 2, 1).to(self.device)
        self.linear2 = nn.Linear(8, 1).to(self.device)
    
    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(self.device)
            gate_inputs = gate_inputs.to(self.device)
            # print(f'\ntemperature{temperature.device}')
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph
    
    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """ 
        Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def _edge_prob_mat(self, graph):
        # The number of nodes in the graph: node_num
        # the number of features in the graph: feature_num
        graph = graph.to(self.device)
        x = self.linear(graph.x).to(self.device)
        f1 = x.unsqueeze(1).repeat(1, 116, 1).view(-1, self.feature_size)
        f2 = x.unsqueeze(0).repeat(116, 1, 1).view(-1, self.feature_size)
        f12self = torch.cat([f1, f2], dim=-1)
        f12self = F.sigmoid(self.linear2(F.sigmoid(self.linear1(f12self))))
        mask_sigmoid = f12self.reshape(116, 116)
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edgemask = sym_mask[graph.edge_index[0], graph.edge_index[1]]
        edgemask = self._sample_graph(edgemask, temperature=0.5, bias=0.0, training=self.training)
        return edgemask
    
    

    def forward(self, graph):
        subgraph = graph.to(self.device)
        edge_prob_matrix = self._edge_prob_mat(subgraph)
        # print(edge_prob_matrix.shape)
        subgraph.attr = edge_prob_matrix

        pos_penalty = edge_prob_matrix.var()
        return subgraph, pos_penalty