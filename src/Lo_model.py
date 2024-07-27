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
from Fun_attention3 import Funattention
class Lo_model(nn.Module):
    def __init__(self, args):
        super(Lo_model, self).__init__()
        set_seed()

        self.dropb1 = nn.Dropout(p=0.2)
        self.dropb2 = nn.Dropout(p=0.2)
        self.dropc1 = nn.Dropout(p=0.2)
        self.dropc2 = nn.Dropout(p=0.2)
        self.dropm1 = nn.Dropout(p=0.2)
        self.dropm2 = nn.Dropout(p=0.2)

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)

        self.FunAtt = Funattention(args)

        self.projectio_bp = nn.Linear(args.projection1_l, args.projection2_l)
        self.projectio_bp1 = nn.Linear(args.projection2_l, args.projection3_l)

        self.projectio_cc = nn.Linear(args.projection1_l, args.projection2_l)
        self.projectio_cc1 = nn.Linear(args.projection2_l, args.projection3_l)

        self.projectio_mf = nn.Linear(args.projection1_l, args.projection2_l)
        self.projectio_mf1 = nn.Linear(args.projection2_l, args.projection3_l)

        self.MLP1 = nn.Linear(args.mlpin_dim_LO, args.mlp1_dim_LO)
        self.MLP2 = nn.Linear(args.mlp1_dim_LO, args.mlp2_dim_LO)
        self.MLP3 = nn.Linear(args.mlp2_dim_LO, args.mlp3_dim_LO)


        self.pool = pyg_nn.global_mean_pool
        self.max_pool = pyg_nn.global_max_pool

        self.st_pool_LO = SAGPool(args.nhid_LO, ratio=args.pooling_ratio_LO)


    def res_para(self):
        self.GT_res.res_para()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()
        self.MLP3.reset_parameters()


    def forward(self,data_G, Fea_BP, fea_CC, fea_MF):
        batch, edge_index = data_G.batch, data_G.edge_index
        BP_GO1 = self.dropb1(self.projectio_bp(Fea_BP))
        BP_GO1 = F.relu(BP_GO1)
        BP_GO = self.dropb2(self.projectio_bp1(BP_GO1))
        BP_GO = F.relu(BP_GO)
        CC_GO1 = self.dropc1(self.projectio_bp(fea_CC))
        CC_GO1 = F.relu(CC_GO1)
        CC_GO = self.dropc2(self.projectio_bp1(CC_GO1))
        CC_GO = F.relu(CC_GO)
        MF_GO1 = self.dropm1(self.projectio_bp(fea_MF))
        MF_GO1 = F.relu(MF_GO1)
        MF_GO = self.dropm2(self.projectio_bp1(MF_GO1))
        MF_GO = F.relu(MF_GO)
        Funrep = self.FunAtt(BP_GO, CC_GO, MF_GO)
        fea_st, score = self.st_pool_LO(Funrep, edge_index, None, batch)
        fea_LO = self.MLP1(fea_st)
        fea_LO1 = self.drop1(fea_LO)
        LO_output = F.sigmoid(fea_LO1)
        return LO_output, fea_LO1, Funrep


