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
from GOmodel import  GO_model
from Lo_model import Lo_model
from train import train
class MaltiTask_model(nn.Module):
    def __init__(self, args):
        super(MaltiTask_model, self).__init__()
        set_seed()
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)

        self.Funmodel = GO_model(args)
        self.Lomodel = Lo_model(args)



    def forward(self,data_G, data_edge):
        BP_out, CC_out, MF_out, BP_AH, CC_AH, MF_AH, Fea_BP, Fea_CC, Fea_MF,feature_bp,feature_cc,feature_mf,fea_GO,res_fea = self.Funmodel(data_G, data_edge)

        LO_output, fea_st,Funrep = self.Lomodel(data_G, Fea_BP,Fea_CC, Fea_MF)
        # print("Main model0000000: Inside GO_model, fea_GO.requires_grad:", Fea_CC.requires_grad, Fea_BP.requires_grad, Fea_MF.requires_grad)
        # Fea_CC.retain_grad()
        # Fea_BP.retain_grad()
        # Fea_MF.retain_grad()
        return BP_out, CC_out, MF_out, BP_AH, CC_AH, MF_AH, LO_output, fea_st, feature_bp,feature_cc,feature_mf,Funrep, fea_GO, res_fea,Fea_BP, Fea_CC, Fea_MF


