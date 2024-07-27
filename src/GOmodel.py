import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from GT import GraghTransformer

from GAE import GAE
from self_attention_pooling import SAGPool
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
class GO_model(nn.Module):
    def __init__(self, args):
        super(GO_model, self).__init__()
        set_seed()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)

        self.GT_res = GraghTransformer(args)

        self.GAE_BP = GAE(args)
        self.GAE_CC = GAE(args)
        self.GAE_MF = GAE(args)

        self.projectio_bp = nn.Linear(args.projection1, args.projection2)
        self.projectio_bp1 = nn.Linear(args.projection2, args.projection3)

        self.projectio_cc = nn.Linear(args.projection1, args.projection2)
        self.projectio_cc1 = nn.Linear(args.projection2, args.projection3)

        self.projectio_mf = nn.Linear(args.projection1, args.projection2)
        self.projectio_mf1 = nn.Linear(args.projection2, args.projection3)


        self.MLP1_cc = nn.Linear(args.mlpin_dim_CC, args.mlp1_dim_CC)
        self.MLP2_cc = nn.Linear(args.mlp1_dim_CC, args.mlp2_dim_CC)
        self.MLP3_cc = nn.Linear(args.mlp2_dim_CC, args.mlp3_dim_CC)

        self.MLP1_bp = nn.Linear(args.mlpin_dim_BP, args.mlp1_dim_BP)
        self.MLP2_bp = nn.Linear(args.mlp1_dim_BP, args.mlp2_dim_BP)
        self.MLP3_bp = nn.Linear(args.mlp2_dim_BP, args.mlp3_dim_BP)

        self.MLP1_mf = nn.Linear(args.mlpin_dim_MF, args.mlp1_dim_MF)
        self.MLP2_mf = nn.Linear(args.mlp1_dim_MF, args.mlp2_dim_MF)
        self.MLP3_mf = nn.Linear(args.mlp2_dim_MF, args.mlp3_dim_MF)


        self.pool = pyg_nn.global_mean_pool
        self.max_pool = pyg_nn.global_max_pool

        self.st_pool_BP = SAGPool(args.nhid, ratio=args.pooling_ratio)
        self.st_pool_CC = SAGPool(args.nhid, ratio=args.pooling_ratio)
        self.st_pool_MF = SAGPool(args.nhid, ratio=args.pooling_ratio)

    def res_para(self):
        self.GT_res.res_para()
        self.MLP1_cc.reset_parameters()
        self.MLP2_cc.reset_parameters()
        self.MLP3_cc.reset_parameters()


    def forward(self,data_G, data_edge):

        res_fea, edge_index, batch = data_G.x, data_G.edge_index, data_G.batch
        edge_fea = data_edge.x
        fea_GO = self.GT_res(res_fea, edge_fea, edge_index)
        BP_GO1 = F.dropout(self.projectio_bp(fea_GO))
        BP_GO1 = F.leaky_relu(BP_GO1)
        BP_GO = F.dropout(self.projectio_bp1(BP_GO1))
        BP_GO = F.leaky_relu(BP_GO)
        CC_GO1 = F.dropout(self.projectio_bp(fea_GO))
        CC_GO1 = F.leaky_relu(CC_GO1)
        CC_GO = F.dropout(self.projectio_bp1(CC_GO1))
        CC_GO = F.leaky_relu(CC_GO)
        MF_GO1 = F.dropout(self.projectio_bp(fea_GO))
        MF_GO1 = F.leaky_relu(MF_GO1)
        MF_GO = F.dropout(self.projectio_bp1(MF_GO1))
        MF_GO = F.leaky_relu(MF_GO)
        BP_AH, Fea_BP = self.GAE_BP(BP_GO, edge_index, batch)
        CC_AH, Fea_CC = self.GAE_CC(CC_GO, edge_index, batch)
        MF_AH, Fea_MF = self.GAE_MF(MF_GO, edge_index, batch)

        fea_BP,score_bp = self.st_pool_BP(Fea_BP, edge_index, None, batch)
        fea_CC,score_cc = self.st_pool_CC(Fea_CC, edge_index, None,  batch)
        fea_MF,score_mf = self.st_pool_MF(Fea_MF, edge_index, None, batch)
        feature_bp = self.MLP1_bp(fea_BP)
        feature_bp1 = self.drop1(feature_bp)
        BP_out = F.sigmoid(feature_bp1)
        feature_cc = self.MLP1_cc(fea_CC)
        feature_cc1 = self.drop1(feature_cc)
        CC_out = F.sigmoid(feature_cc1)
        feature_mf = self.MLP1_mf(fea_MF)
        feature_mf1 = self.drop1(feature_mf)
        MF_out = F.sigmoid(feature_mf1)
        return BP_out, CC_out, MF_out, BP_AH, CC_AH, MF_AH, Fea_BP, Fea_CC, Fea_MF, feature_bp,feature_cc,feature_mf,fea_GO,res_fea


