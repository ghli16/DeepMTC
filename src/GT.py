
import torch as th
import torch.nn.functional as F
import copy
import numpy as np
import random
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut, to_dense_batch
from torch_geometric.nn import MetaLayer
from torch import nn
import pandas as pd
from utils import *


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True):
        super(MultiHeadAttentionLayer, self).__init__()
        set_seed()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection):
        row, col = edge_index
        e_out = None
        # Compute attention scores
        alpha = node_feats_k[row] * node_feats_q[col]
        # Scale and clip attention scores
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        # Use available edge features to modify the attention scores
        alpha = alpha * edge_feats_projection
        # Copy edge features as e_out to be passed to edge_feats_MLP
        if self.update_edge_feats:
            e_out = alpha

        # Apply softmax to attention scores, followed by clipping
        alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0, 5.0))
        # Send weighted values to target nodes
        wV = scatter_add(node_feats_v[row] * alphax, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
        return wV, z, e_out

    def forward(self, x, edge_attr, edge_index):
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v,
                                                edge_feats_projection)

        h_out = wV / (z + th.full_like(z, 1e-6))
        return h_out, e_out


class GraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()
        set_seed()
        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

    def run_gt_layer(self, edge_index, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, edge_attn_out = self.mha_module(node_feats, edge_feats, edge_index)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return node_feats, edge_feats

    def forward(self, edge_index, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats, edge_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()
        set_seed()
        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers



        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)


    def run_gt_layer(self, edge_index, node_feats, edge_feats):
        set_seed()
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        # edge_feats = self.conformation_module(edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, _ = self.mha_module(node_feats, edge_feats, edge_index)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLP for node features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return node representations
        return node_feats

    def forward(self, edge_index, node_feats, edge_feats):

        node_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
        return node_feats


class GraghTransformer(nn.Module):

    def __init__(self, args):
        super(GraghTransformer, self).__init__()
        set_seed()
        self.in_channels = args.in_channels
        self.edge_features = args.edge_features
        self.num_hidden_channels = args.num_hidden_channels
        self.num_attention_heads = args.num_attention_heads
        self.dropout_rate = args.dropout_rate
        self.num_layers = args.num_layers
        self.transformer_residual = args.transformer_residual
        self.norm_to_apply = args.norm_to_apply

        self.activ_fn = nn.SiLU()


        self.node_encoder = nn.Linear(self.in_channels, self.num_hidden_channels)
        self.edge_encoder = nn.Linear(self.edge_features, self.num_hidden_channels)

        num_intermediate_layers = max(0, self.num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=self.num_hidden_channels,
            activ_fn=self.activ_fn,
            residual=self.transformer_residual,
            num_attention_heads=self.num_attention_heads,
            norm_to_apply=self.norm_to_apply,
            dropout_rate=self.dropout_rate,
            num_layers=self.num_layers) for _ in range(num_intermediate_layers)]
        if self.num_layers > 0:
            gt_block_modules.append(FinalGraphTransformerModule(self.num_hidden_channels, self.activ_fn, self.transformer_residual, self.num_attention_heads, self.norm_to_apply, self.dropout_rate, self.num_layers))
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, node_s, edge_s, edge_index):
        node_feats = self.node_encoder(node_s)
        edge_feats = self.edge_encoder(edge_s)

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(edge_index, node_feats, edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        node_feats = self.gt_block[-1](edge_index, node_feats, edge_feats)
        # return node_feats
        return node_feats