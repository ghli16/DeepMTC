from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import torch_geometric.nn as pyg_nn
from utils import  args, set_seed
class GAE(torch.nn.Module):
    def __init__(self, args):
        super(GAE, self).__init__()
        set_seed()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.activate = F.leaky_relu
        self.node_features = args.num_hidden_channels
        # self.node_features = args.in_channels
        self.gcn1_features = args.gcn1_features
        self.gcn2_features = args.gcn2_features
        self.gcn3_features = args.gcn3_features

        self.GCN1 = GCNConv(self.node_features, self.gcn1_features)
        self.GCN2 = GCNConv(self.gcn1_features, self.gcn2_features)
        self.GCN3 = GCNConv(self.gcn2_features, self.gcn3_features)


    def forward(self, x, edge_index, batch):
        x = self.activate(self.GCN1(x, edge_index))
        x = self.drop1(x)
        x = self.activate(self.GCN2(x, edge_index))
        x = self.drop2(x)
        # x = self.activate(self.GCN3(x, edge_index))
        x1 = torch.nn.functional.sigmoid(torch.matmul(x, x.T))
        return x1, x