import argparse
import torch.nn as nn
import pandas as pd
import networkx as nx
import numpy as np
import random

import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
import os
from sklearn import metrics
import csv
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int, help='The training epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--projection1', type=int, default=512, help='projection.')
parser.add_argument('--projection2', type=int, default=512, help='projection.')
parser.add_argument('--projection3', type=int, default=512, help='projection.')


parser.add_argument('--projection1_l', type=int, default=128, help='projection.')
parser.add_argument('--projection2_l', type=int, default=64, help='projection.')
parser.add_argument('--projection3_l', type=int, default=128, help='projection.')



# pooling
parser.add_argument('--nhid', default=128, type=int, help='channel')
parser.add_argument('--pooling_ratio', default=0.5, type=float, help='pooling_ratio')

parser.add_argument('--nhid_LO', default=128, type=int, help='channel')
parser.add_argument('--pooling_ratio_LO', default=0.4, type=float, help='pooling_ratio')





# GT_para
parser.add_argument('--in_channels', default=1280, type=int, help='initial feature dimension.')
parser.add_argument('--edge_features', default=20, type=int, help='initial edge feature dimension.')
parser.add_argument('--num_hidden_channels', default=512, type=int, help='hidden feature dimension.') #in_dim
parser.add_argument('--num_attention_heads', default=8, type=int, help='heads number.')
parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout_rate.')
parser.add_argument('--num_layers', default=2, type=int, help='GraphTransformer layers.')
parser.add_argument('--transformer_residual', default=True, type=bool, help='transformer_residual.')
parser.add_argument('--norm_to_apply', default='batch', type=str, help='norm_to_apply.')

# attention
parser.add_argument('--hidden_dim', default=128, type=int, help='hidden_dim.')
parser.add_argument('--fout_dim', default=128, type=int, help='fout_dim.')
parser.add_argument('--num_heads', default=4, type=int, help='num_heads.')
parser.add_argument('--layer_norm', default=True, type=bool, help='layer_norm.')
parser.add_argument('--batch_norm', default=False, type=bool, help='batch_norm.')
parser.add_argument('--residual', default=True, type=bool, help='residual.')

# GAE
parser.add_argument('--gcn1_features', default=256, type=int, help='encode1 sequences features.')
parser.add_argument('--gcn2_features', default=128, type=int, help='encode2 sequences features.')
parser.add_argument('--gcn3_features', default=128, type=int, help='decode1 sequences features.')


parser.add_argument('--mlpin_dim_BP', type=int, default=256, help='mlpini_dim.')
parser.add_argument('--mlp1_dim_BP', type=int, default=119, help='mlpmid_dim.')
parser.add_argument('--mlp2_dim_BP', type=int, default=119, help='mlpini_dim.')
parser.add_argument('--mlp3_dim_BP', type=int, default=119, help='mlpini_dim.')

parser.add_argument('--mlpin_dim_CC', type=int, default=256, help='mlpini_dim.')
parser.add_argument('--mlp1_dim_CC', type=int, default=101, help='mlpmid_dim.')
parser.add_argument('--mlp2_dim_CC', type=int, default=101, help='mlpini_dim.')
parser.add_argument('--mlp3_dim_CC', type=int, default=101, help='mlpini_dim.')

parser.add_argument('--mlpin_dim_MF', type=int, default=256, help='mlpini_dim.')
parser.add_argument('--mlp1_dim_MF', type=int, default=69, help='mlpmid_dim.')
parser.add_argument('--mlp2_dim_MF', type=int, default=69, help='mlpini_dim.')
parser.add_argument('--mlp3_dim_MF', type=int, default=69, help='mlpini_dim.')

parser.add_argument('--mlpin_dim_LO', type=int, default=256, help='mlpini_dim.')
parser.add_argument('--mlp1_dim_LO', type=int, default=10, help='mlpmid_dim.')#64
parser.add_argument('--mlp2_dim_LO', type=int, default=10, help='mlpini_dim.')#32  6
parser.add_argument('--mlp3_dim_LO', type=int, default=10, help='mlpini_dim.')

# parser.add_argument('--mlp4_dim', type=float, default=1, help='mlpdin_dim.')
args = parser.parse_args()


def set_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def write_csv_file(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)




classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]






def calculate_performance(actual, pred_prob, threshold=0.5, average='micro'):
    pred_lable = []
    actual_label = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int_)
        eachline = eachline.tolist()

        pred_lable.append(eachline)
    for l in range(len(actual)):
        eachline = (np.array(actual[l])).astype(np.int_)
        eachline = eachline.tolist()
        actual_label.append(eachline)
    f_score = f1_score(actual_label, pred_lable, average=average)
    recall = recall_score(actual_label, pred_lable, average=average)
    precision = precision_score(actual_label,  pred_lable, average=average, zero_division=1.0)
    return f_score, precision, recall




def edge_index():
    path = "../Dataset/Edge_index/graph_LO(45).csv"
    matrix_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
    return matrix_list

def get_data(path):
    matrix_list = []

    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
                        num_rows = None
    return matrix_list


# edge_index:
def node_feature():
    path1 = "../Dataset/Node_feature/Front_extrac_feature.csv"
    path2 = "../Dataset/Node_feature/Mid_extrac_feature.csv"
    path3 = "../Dataset/Node_feature/End_extrac_feature.csv"
    list = get_data(path1) + get_data(path2) + get_data(path3)
    merged_list = list
    return merged_list

def edge_feature():
    path1 = "../Dataset/Edge_feature/Edge_fea(4.5)_font.csv"
    path2 = "../Dataset/Edge_feature/Edge_fea(4.5)_mid.csv"
    path3 = "../Dataset/Edge_feature/Edge_fea(4.5)_end.csv"
    list = get_data(path1) + get_data(path2) + get_data(path3)
    merged_list = list
    return merged_list

def get_go_label(path):
    # 读取CSV文件
    df = pd.read_csv(path, header=None, dtype=int)  # 不使用默认的列名
    matrix = df.values
    return  matrix

def go_lb():
    path1 = "../Dataset/True_GO_Label/BP_La.csv"
    path2 = "../Dataset/True_GO_Label/CC_La.csv"
    path3 = "../Dataset/True_GO_Label/MF_La.csv"
    BP = get_go_label(path1)
    CC = get_go_label(path2)
    MF = get_go_label(path3)
    return BP, CC, MF
def read_csv_file(filename):
    data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data



def lo_lb1():
    pathl= "../Dataset/LO_label/multi_label_subcellular_locations（one_hot）.csv"
    LO = get_go_label(pathl)
    return LO
def adjacency_matrix_to_edge_index(adjacency_matrix):
    edge_index = []
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                edge_index.append((i, j))
    return edge_index

def data_load():
    start_time = time.time()  # 记录开始时间
    set_seed()
    Node_fea = node_feature()
    Node_fea = (Node_fea[1:5] + Node_fea[6:50] + Node_fea[51:247] + Node_fea[248:746] + Node_fea[747:836] + Node_fea[837:1013] + Node_fea[1014:1147] + Node_fea[1148:1364] + Node_fea[1365:1544] + Node_fea[1545:1569] + Node_fea[1570:1636] + Node_fea[1637:1651] + Node_fea[1652:1744] + Node_fea[1745:1784] + Node_fea[1785:1887]+ Node_fea[1888:1925] + Node_fea[1926:1938] + Node_fea[1939:2142] + Node_fea[2143:2301] +
                Node_fea[2302:2388] + Node_fea[2389:2551] + Node_fea[2552:2668] + Node_fea[2669:2722] + Node_fea[2723:3135] + Node_fea[3136:3200] + Node_fea[3201:3207] + Node_fea[3208:4149] + Node_fea[4150:4155] + Node_fea[4156:4176] + Node_fea[4177:4215] + Node_fea[4216:4217] + Node_fea[4218:4225] + Node_fea[4226:4236] + Node_fea[4237:4238] + Node_fea[4239:4241] + Node_fea[4242:4243] + Node_fea[4244:4252] + Node_fea[4253:4257] + Node_fea[4258:4273] +
                Node_fea[4274:4275] + Node_fea[4276:4279] + Node_fea[4281:4296] + Node_fea[4297:4304] + Node_fea[4305:4311] + Node_fea[4312:4316] + Node_fea[4317:4321] + Node_fea[4322:4325] + Node_fea[4326:4349] + Node_fea[4350:4382] + Node_fea[4383:4397] + Node_fea[4398:4408] + Node_fea[4409:4420] + Node_fea[4421:4449] + Node_fea[4450:4456] + Node_fea[4459:4483] + Node_fea[4484:4489] + Node_fea[4490:4491] + Node_fea[4492:4495] + Node_fea[4496:4544] +
                Node_fea[4545:4564] + Node_fea[4565:4834] + Node_fea[4835:5111] + Node_fea[5112:5147] + Node_fea[5148:5176] + Node_fea[5177:5214] +Node_fea[5215:5221] + Node_fea[5222:5277] + Node_fea[5278:5389] + Node_fea[5390:5424] + Node_fea[5425:5472] + Node_fea[5474:5483] + Node_fea[5484:5516] + Node_fea[5517:5522] + Node_fea[5523:5550] + Node_fea[5551:5576] + Node_fea[5577:5580] + Node_fea[5581:5643] + Node_fea[5644:5690] + Node_fea[5691:5711] +
                Node_fea[5712:5762] + Node_fea[5763:5772] + Node_fea[5773:5782] + Node_fea[5783:5796] + Node_fea[5797:5805] + Node_fea[5806:5817] + Node_fea[5818:5821] + Node_fea[5822:5827] + Node_fea[5828:5936] + Node_fea[5937:5938] + Node_fea[5939:5947] + Node_fea[5948:5974] + Node_fea[5975:5977] + Node_fea[5978:5984] + Node_fea[5985:6019] + Node_fea[6020:6058] + Node_fea[6059:6060] + Node_fea[6061:6110] + Node_fea[6111:6136] + Node_fea[6137:6143] +
                Node_fea[6144:6145] + Node_fea[6146:6148] + Node_fea[6151:6153] + Node_fea[6154:6168] + Node_fea[6170:])
    Edge_fea = edge_feature()
    Edge_fea = (Edge_fea[1:5] + Edge_fea[6:50] + Edge_fea[51:247] + Edge_fea[248:746] + Edge_fea[747:836] + Edge_fea[837:1013] + Edge_fea[1014:1147] + Edge_fea[1148:1364] + Edge_fea[1365:1544] + Edge_fea[1545:1569] + Edge_fea[1570:1636] + Edge_fea[1637:1651] + Edge_fea[1652:1744] + Edge_fea[1745:1784] + Edge_fea[1785:1887]+ Edge_fea[1888:1925] + Edge_fea[1926:1938] + Edge_fea[1939:2142] + Edge_fea[2143:2301] +
                Edge_fea[2302:2388] + Edge_fea[2389:2551] + Edge_fea[2552:2668] + Edge_fea[2669:2722] + Edge_fea[2723:3135] + Edge_fea[3136:3200] + Edge_fea[3201:3207] + Edge_fea[3208:4149] + Edge_fea[4150:4155] + Edge_fea[4156:4176] + Edge_fea[4177:4215] + Edge_fea[4216:4217] + Edge_fea[4218:4225] + Edge_fea[4226:4236] + Edge_fea[4237:4238] + Edge_fea[4239:4241] + Edge_fea[4242:4243] + Edge_fea[4244:4252] + Edge_fea[4253:4257] + Edge_fea[4258:4273] +
                Edge_fea[4274:4275] + Edge_fea[4276:4279] + Edge_fea[4281:4296] + Edge_fea[4297:4304] + Edge_fea[4305:4311] + Edge_fea[4312:4316] + Edge_fea[4317:4321] + Edge_fea[4322:4325] + Edge_fea[4326:4349] + Edge_fea[4350:4382] + Edge_fea[4383:4397] + Edge_fea[4398:4408] + Edge_fea[4409:4420] + Edge_fea[4421:4449] + Edge_fea[4450:4456] + Edge_fea[4459:4483] + Edge_fea[4484:4489] + Edge_fea[4490:4491] + Edge_fea[4492:4495] + Edge_fea[4496:4544] +
                Edge_fea[4545:4564] + Edge_fea[4565:4834] + Edge_fea[4835:5111] + Edge_fea[5112:5147] + Edge_fea[5148:5176] + Edge_fea[5177:5214] +Edge_fea[5215:5221] + Edge_fea[5222:5277] + Edge_fea[5278:5389] + Edge_fea[5390:5424] + Edge_fea[5425:5472] + Edge_fea[5474:5483] + Edge_fea[5484:5516] + Edge_fea[5517:5522] + Edge_fea[5523:5550] + Edge_fea[5551:5576] + Edge_fea[5577:5580] + Edge_fea[5581:5643] + Edge_fea[5644:5690] + Edge_fea[5691:5711] +
                Edge_fea[5712:5762] + Edge_fea[5763:5772] + Edge_fea[5773:5782] + Edge_fea[5783:5796] + Edge_fea[5797:5805] + Edge_fea[5806:5817] + Edge_fea[5818:5821] + Edge_fea[5822:5827] + Edge_fea[5828:5936] + Edge_fea[5937:5938] + Edge_fea[5939:5947] + Edge_fea[5948:5974] + Edge_fea[5975:5977] + Edge_fea[5978:5984] + Edge_fea[5985:6019] + Edge_fea[6020:6058] + Edge_fea[6059:6060] + Edge_fea[6061:6110] + Edge_fea[6111:6136] + Edge_fea[6137:6143] +
                Edge_fea[6144:6145] + Edge_fea[6146:6148] + Edge_fea[6151:6153] + Edge_fea[6154:6168] + Edge_fea[6170:])


    Edge_index = edge_index()

    Edge_index = (Edge_index[1:5] + Edge_index[6:50] + Edge_index[51:247] + Edge_index[248:746] + Edge_index[747:836] + Edge_index[837:1013] + Edge_index[1014:1147] + Edge_index[1148:1364] + Edge_index[1365:1544] + Edge_index[1545:1569] + Edge_index[1570:1636] + Edge_index[1637:1651] + Edge_index[1652:1744] + Edge_index[1745:1784] + Edge_index[1785:1887]+ Edge_index[1888:1925] + Edge_index[1926:1938] + Edge_index[1939:2142] + Edge_index[2143:2301] +
                Edge_index[2302:2388] + Edge_index[2389:2551] + Edge_index[2552:2668] + Edge_index[2669:2722] + Edge_index[2723:3135] + Edge_index[3136:3200] + Edge_index[3201:3207] + Edge_index[3208:4149] + Edge_index[4150:4155] + Edge_index[4156:4176] + Edge_index[4177:4215] + Edge_index[4216:4217] + Edge_index[4218:4225] + Edge_index[4226:4236] + Edge_index[4237:4238] + Edge_index[4239:4241] + Edge_index[4242:4243] + Edge_index[4244:4252] + Edge_index[4253:4257] + Edge_index[4258:4273] +
                Edge_index[4274:4275] + Edge_index[4276:4279] + Edge_index[4281:4296] + Edge_index[4297:4304] + Edge_index[4305:4311] + Edge_index[4312:4316] + Edge_index[4317:4321] + Edge_index[4322:4325] + Edge_index[4326:4349] + Edge_index[4350:4382] + Edge_index[4383:4397] + Edge_index[4398:4408] + Edge_index[4409:4420] + Edge_index[4421:4449] + Edge_index[4450:4456] + Edge_index[4459:4483] + Edge_index[4484:4489] + Edge_index[4490:4491] + Edge_index[4492:4495] + Edge_index[4496:4544] +
                Edge_index[4545:4564] + Edge_index[4565:4834] + Edge_index[4835:5111] + Edge_index[5112:5147] + Edge_index[5148:5176] + Edge_index[5177:5214] +Edge_index[5215:5221] + Edge_index[5222:5277] + Edge_index[5278:5389] + Edge_index[5390:5424] + Edge_index[5425:5472] + Edge_index[5474:5483] + Edge_index[5484:5516] + Edge_index[5517:5522] + Edge_index[5523:5550] + Edge_index[5551:5576] + Edge_index[5577:5580] + Edge_index[5581:5643] + Edge_index[5644:5690] + Edge_index[5691:5711] +
                Edge_index[5712:5762] + Edge_index[5763:5772] + Edge_index[5773:5782] + Edge_index[5783:5796] + Edge_index[5797:5805] + Edge_index[5806:5817] + Edge_index[5818:5821] + Edge_index[5822:5827] + Edge_index[5828:5936] + Edge_index[5937:5938] + Edge_index[5939:5947] + Edge_index[5948:5974] + Edge_index[5975:5977] + Edge_index[5978:5984] + Edge_index[5985:6019] + Edge_index[6020:6058] + Edge_index[6059:6060] + Edge_index[6061:6110] + Edge_index[6111:6136] + Edge_index[6137:6143] +
                Edge_index[6144:6145] + Edge_index[6146:6148] + Edge_index[6151:6153] + Edge_index[6154:6168] + Edge_index[6170:])


    numbers = [
        0, 5, 50, 247, 746, 836, 1013, 1147, 1364, 1544,
        1569, 1636, 1651, 1744, 1784, 1887, 1925, 1938, 2142,
        2301, 2388, 2551, 2668, 2722, 3135, 3200, 3207, 4149,
        4155, 4176, 4215, 4217, 4225, 4236, 4238, 4241, 4243,
        4252, 4257, 4273, 4275, 4279, 4280, 4296, 4304, 4311,
        4316, 4321, 4325, 4349, 4382, 4397, 4408, 4420, 4449,
        4456, 4457, 4458, 4483, 4489, 4491, 4495, 4544, 4564,
        4834, 5111, 5147, 5176, 5214, 5221, 5277, 5389, 5424,
        5472, 5473, 5483, 5516, 5522, 5550, 5576, 5580, 5643,
        5690, 5711, 5762, 5772, 5782, 5796, 5805, 5817, 5821,
        5827, 5936, 5938, 5947, 5974, 5977, 5984, 6019, 6058,
        6060, 6110, 6136, 6143, 6145, 6148, 6149, 6150, 6153,
        6168, 6169
    ]

    BP_lb, CC_lb, MF_lb = go_lb()
    BP_lb = np.delete(BP_lb, numbers, axis=0)
    CC_lb = np.delete(CC_lb, numbers, axis=0)
    MF_lb = np.delete(MF_lb, numbers, axis=0)
    LO_LB = lo_lb1()
    LO_lb = np.delete(LO_LB, numbers, axis=0)

    filename = '../Dataset/HM_ID.csv'  # 将文件名替换为你的CSV文件名

    # 调用函数读取CSV文件
    csv_data = read_csv_file(filename)
    csv_data = (csv_data[1:5] + csv_data[6:50] + csv_data[51:247] + csv_data[248:746] + csv_data[747:836] + csv_data[
                                                                                                            837:1013] + csv_data[
                                                                                                                        1014:1147] + csv_data[
                                                                                                                                     1148:1364] + csv_data[
                                                                                                                                                  1365:1544] + csv_data[
                                                                                                                                                               1545:1569] + csv_data[
                                                                                                                                                                            1570:1636] + csv_data[
                                                                                                                                                                                         1637:1651] + csv_data[
                                                                                                                                                                                                      1652:1744] + csv_data[
                                                                                                                                                                                                                   1745:1784] + csv_data[
                                                                                                                                                                                                                                1785:1887] + csv_data[
                                                                                                                                                                                                                                             1888:1925] + csv_data[
                                                                                                                                                                                                                                                          1926:1938] + csv_data[
                                                                                                                                                                                                                                                                       1939:2142] + csv_data[
                                                                                                                                                                                                                                                                                    2143:2301] +
                csv_data[2302:2388] + csv_data[2389:2551] + csv_data[2552:2668] + csv_data[2669:2722] + csv_data[
                                                                                                        2723:3135] + csv_data[
                                                                                                                     3136:3200] + csv_data[
                                                                                                                                  3201:3207] + csv_data[
                                                                                                                                               3208:4149] + csv_data[
                                                                                                                                                            4150:4155] + csv_data[
                                                                                                                                                                         4156:4176] + csv_data[
                                                                                                                                                                                      4177:4215] + csv_data[
                                                                                                                                                                                                   4216:4217] + csv_data[
                                                                                                                                                                                                                4218:4225] + csv_data[
                                                                                                                                                                                                                             4226:4236] + csv_data[
                                                                                                                                                                                                                                          4237:4238] + csv_data[
                                                                                                                                                                                                                                                       4239:4241] + csv_data[
                                                                                                                                                                                                                                                                    4242:4243] + csv_data[
                                                                                                                                                                                                                                                                                 4244:4252] + csv_data[
                                                                                                                                                                                                                                                                                              4253:4257] + csv_data[
                                                                                                                                                                                                                                                                                                           4258:4273] +
                csv_data[4274:4275] + csv_data[4276:4279] + csv_data[4281:4296] + csv_data[4297:4304] + csv_data[
                                                                                                        4305:4311] + csv_data[
                                                                                                                     4312:4316] + csv_data[
                                                                                                                                  4317:4321] + csv_data[
                                                                                                                                               4322:4325] + csv_data[
                                                                                                                                                            4326:4349] + csv_data[
                                                                                                                                                                         4350:4382] + csv_data[
                                                                                                                                                                                      4383:4397] + csv_data[
                                                                                                                                                                                                   4398:4408] + csv_data[
                                                                                                                                                                                                                4409:4420] + csv_data[
                                                                                                                                                                                                                             4421:4449] + csv_data[
                                                                                                                                                                                                                                          4450:4456] + csv_data[
                                                                                                                                                                                                                                                       4459:4483] + csv_data[
                                                                                                                                                                                                                                                                    4484:4489] + csv_data[
                                                                                                                                                                                                                                                                                 4490:4491] + csv_data[
                                                                                                                                                                                                                                                                                              4492:4495] + csv_data[
                                                                                                                                                                                                                                                                                                           4496:4544] +
                csv_data[4545:4564] + csv_data[4565:4834] + csv_data[4835:5111] + csv_data[5112:5147] + csv_data[
                                                                                                        5148:5176] + csv_data[
                                                                                                                     5177:5214] + csv_data[
                                                                                                                                  5215:5221] + csv_data[
                                                                                                                                               5222:5277] + csv_data[
                                                                                                                                                            5278:5389] + csv_data[
                                                                                                                                                                         5390:5424] + csv_data[
                                                                                                                                                                                      5425:5472] + csv_data[
                                                                                                                                                                                                   5474:5483] + csv_data[
                                                                                                                                                                                                                5484:5516] + csv_data[
                                                                                                                                                                                                                             5517:5522] + csv_data[
                                                                                                                                                                                                                                          5523:5550] + csv_data[
                                                                                                                                                                                                                                                       5551:5576] + csv_data[
                                                                                                                                                                                                                                                                    5577:5580] + csv_data[
                                                                                                                                                                                                                                                                                 5581:5643] + csv_data[
                                                                                                                                                                                                                                                                                              5644:5690] + csv_data[
                                                                                                                                                                                                                                                                                                           5691:5711] +
                csv_data[5712:5762] + csv_data[5763:5772] + csv_data[5773:5782] + csv_data[5783:5796] + csv_data[
                                                                                                        5797:5805] + csv_data[
                                                                                                                     5806:5817] + csv_data[
                                                                                                                                  5818:5821] + csv_data[
                                                                                                                                               5822:5827] + csv_data[
                                                                                                                                                            5828:5936] + csv_data[
                                                                                                                                                                         5937:5938] + csv_data[
                                                                                                                                                                                      5939:5947] + csv_data[
                                                                                                                                                                                                   5948:5974] + csv_data[
                                                                                                                                                                                                                5975:5977] + csv_data[
                                                                                                                                                                                                                             5978:5984] + csv_data[
                                                                                                                                                                                                                                          5985:6019] + csv_data[
                                                                                                                                                                                                                                                       6020:6058] + csv_data[
                                                                                                                                                                                                                                                                    6059:6060] + csv_data[
                                                                                                                                                                                                                                                                                 6061:6110] + csv_data[
                                                                                                                                                                                                                                                                                              6111:6136] + csv_data[
                                                                                                                                                                                                                                                                                                           6137:6143] +
                csv_data[6144:6145] + csv_data[6146:6148] + csv_data[6151:6153] + csv_data[6154:6168] + csv_data[6170:])



    data_pro = [Data(x=torch.tensor(Node_fea[i], dtype=torch.float),
                      edge_index=torch.tensor((Edge_index[i]), dtype=torch.long))
                 for i in range(len(Node_fea))]   #LO_lb

    data_Edge_fea = [Data(x=torch.tensor(Edge_fea[i], dtype=torch.float))
                  for i in range(len(Edge_fea))]

    data_LO_label = [Data(y=torch.tensor(LO_lb[i], dtype=torch.float))
                  for i in range(len(LO_lb))]

    data_BP_label = [Data(y=torch.tensor(BP_lb[i], dtype=torch.float))
                  for i in range(len(BP_lb))]

    data_CC_label = [Data(y=torch.tensor(CC_lb[i], dtype=torch.float))
                     for i in range(len(CC_lb))]

    data_MF_label = [Data(y=torch.tensor(MF_lb[i], dtype=torch.float))
                     for i in range(len(MF_lb))]


    set_seed()
    random.shuffle(data_pro)
    random.shuffle(data_Edge_fea)
    random.shuffle(data_LO_label)
    random.shuffle(data_BP_label)
    random.shuffle(data_CC_label)
    random.shuffle(data_MF_label)
    random.shuffle(csv_data)
    write_csv_file(csv_data, '../Dataset/HM_ID_SHUFF.csv')
    end_time = time.time()
    elapsed_time =(end_time - start_time)/60
    print(f"Data creation took {elapsed_time:.2f} minutes")
    return data_pro, data_LO_label, data_Edge_fea, data_BP_label, data_CC_label, data_MF_label
