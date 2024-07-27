from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
import networkx as nx
from utils import *
from torch.utils.data.dataloader import DataLoader

class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        set_seed()
        assert in_dim % num_heads == 0
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.num_heads = num_heads
        self.depth = in_dim // num_heads
        self.out_dim = out_dim


        self.query_linear1 = nn.Linear(in_dim, in_dim)
        self.key_linear1 = nn.Linear(in_dim, in_dim)
        self.value_linear1 = nn.Linear(in_dim, in_dim)
        self.query_linear2 = nn.Linear(in_dim, in_dim)
        self.key_linear2 = nn.Linear(in_dim, in_dim)
        self.value_linear2 = nn.Linear(in_dim, in_dim)
        self.query_linear3 = nn.Linear(in_dim, in_dim)
        self.key_linear3 = nn.Linear(in_dim, in_dim)
        self.value_linear3 = nn.Linear(in_dim, in_dim)

        self.output_linear = nn.Linear(in_dim, out_dim)

    def res_para(self):
        set_seed()
        self.query_linear.reset_parameters()
        self.key_linear.reset_parameters()
        self.value_linear.reset_parameters()
        self.output_linear.reset_parameters()

    def split_heads(self, x, batch_size):
        # reshape input to [batch_size, num_heads, seq_len, depth]
        set_seed()
        x_szie = x.size()[:-1] + (self.num_heads, self.depth)
        x = x.reshape(x_szie)
        # transpose to [batch_size, num_heads, depth, seq_len]
        return x.transpose(-1, -2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q1 = self.query_linear1(query)
        V2 = self.value_linear2(query)
        K3 = self.key_linear3(query)

        K2 = self.key_linear1(key)
        V1 = self.value_linear1(key)
        Q3 = self.query_linear3(key)

        Q2 = self.query_linear2(value)
        K1 = self.key_linear2(value)
        V3 = self.value_linear3(value)


        # Split the inputs into multiple heads
        Q1 = self.split_heads(Q1, batch_size)
        K1 = self.split_heads(K1, batch_size)
        V1 = self.split_heads(V1, batch_size)

        Q2 = self.split_heads(Q2, batch_size)
        K2 = self.split_heads(K2, batch_size)
        V2 = self.split_heads(V2, batch_size)

        Q3 = self.split_heads(Q3, batch_size)
        K3 = self.split_heads(K3, batch_size)
        V3 = self.split_heads(V3, batch_size)

        # Scaled Dot-Product Attention
        scores1 = torch.matmul(Q1, K1.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scores2 = torch.matmul(Q2, K2.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        scores3 = torch.matmul(Q3, K3.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))


        # Apply mask (if necessary)
        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dimension
            scores = scores1.masked_fill(mask == 0, -1e9)

        attention_weights1 = torch.softmax(scores1, dim=1)
        attention_weights2 = torch.softmax(scores2, dim=1)
        attention_weights3 = torch.softmax(scores3, dim=1)


        attention_output1 = torch.matmul(attention_weights1, V1)
        attention_output2 = torch.matmul(attention_weights2, V2)
        attention_output3 = torch.matmul(attention_weights3, V3)
        # attention_output2 = torch.matmul(attention_weights,position)
        # attention_output = torch.cat((attention_output1,attention_output2), dim=1)
        # Merge the heads
        output_size1 = attention_output1.size()[:-2] + (query.size(1),)
        attention_output1 = attention_output1.transpose(-1, -2).reshape((output_size1))

        output_size2 = attention_output2.size()[:-2] + (query.size(1),)
        attention_output2 = attention_output2.transpose(-1, -2).reshape((output_size2))

        output_size3 = attention_output3.size()[:-2] + (query.size(1),)
        attention_output3 = attention_output3.transpose(-1, -2).reshape((output_size3))




        attention_output = (attention_output1 + attention_output2+ attention_output3)/3

        attention_output = F.relu(self.output_linear(attention_output))



        return attention_output



class Funattention(nn.Module):
    def __init__(self, args):
        super(Funattention, self).__init__()
        set_seed()
        self.in_dim = args.gcn2_features
        self.hidden_dim = args.hidden_dim
        self.fout_dim = args.fout_dim
        self.num_heads = args.num_heads
        self.dropout = args.dropout_rate
        self.residual = args.residual
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.attention = MultiheadAttention(self.in_dim, self.hidden_dim, self.num_heads)
        # self.residual_layer1 = nn.Linear(node_features, fout_dim)  #残差
        self.O = nn.Linear(self.hidden_dim, self.fout_dim)
        # self.node_features = node_features



        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(self.fout_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.fout_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(self.fout_dim, self.fout_dim * 2)
        self.FFN_layer2 = nn.Linear(self.fout_dim * 2, self.fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(self.fout_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(self.fout_dim)
    def res_para(self):
        set_seed()
        self.residual_layer1.reset_parameters()
        self.O.reset_parameters()
        self.attention.res_para()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        # self.batch_norm1.reset_parameters()
        # self.batch_norm2.reset_parameters()
        self.FFN_layer1.reset_parameters()
        self.FFN_layer2.reset_parameters()


    def forward(self, BP_fea, CC_fea, MF_fea):

        attn_out = self.attention(BP_fea, CC_fea, MF_fea)

        # attn_out = F.dropout(attn_out, self.dropout, training=self.training)

        # attn_out = F.leaky_relu(self.O(attn_out))
        # if self.layer_norm:
        #     attn_out = self.layer_norm1(attn_out)
        #
        # if self.batch_norm:
        #     attn_out = self.batch_norm1(attn_out)

        h_in2 = attn_out  # for second residual connection

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.relu(attn_out)

        if self.residual:
            attn_out = h_in2 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)
        return attn_out