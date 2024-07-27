import random
import gc
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from sklearn.metrics import hamming_loss, coverage_error,precision_recall_curve
from GT import GraghTransformer
from Main_model import MaltiTask_model
from sklearn.metrics import matthews_corrcoef
import json
import matplotlib.pyplot as plt
import time
from Data_pro import Data_loader
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc


def train():
    set_seed()
    train_data, valid_data, test_data, \
        train_edge, valid_edge, test_edge, \
        train_LO, valid_LO, test_LO, \
        train_BP, valid_BP, test_BP, \
        train_CC, valid_CC, test_CC, \
        train_MF, valid_MF, test_MF, \
        test_data1, test_data2, test_data3, test_data4, test_data5, \
        test_edge1, test_edge2, test_edge3, test_edge4, test_edge5, \
        test_LO1, test_LO2, test_LO3, test_LO4, test_LO5, \
        test_BP1, test_BP2, test_BP3, test_BP4, test_BP5, \
        test_CC1, test_CC2, test_CC3, test_CC4, test_CC5, \
        test_MF1, test_MF2, test_MF3, test_MF4, test_MF5 = Data_loader()

    model = MaltiTask_model(args).cuda()
    patch = '../model_para_42/3fun.pt'
    cpkt = torch.load(patch)
    model.load_state_dict(cpkt)
    y_pred_all_cc = []
    y_pred_all_bp = []
    y_pred_all_mf = []

    real_all_cc = []
    real_all_bp = []
    real_all_mf = []
    lo_pre_label = []
    y_pred_all_lo = []
    real_all_lo = []
    pre_lo = torch.Tensor([]).cuda()
    pre_lop = torch.Tensor([]).cuda()
    true_lo = torch.Tensor([]).cuda()

    true_go_bp = torch.Tensor([]).cuda()
    pre_go_bp = torch.Tensor([]).cuda()
    true_go_cc = torch.Tensor([]).cuda()
    pre_go_cc = torch.Tensor([]).cuda()
    true_go_mf = torch.Tensor([]).cuda()
    pre_go_mf = torch.Tensor([]).cuda()

    model.eval()
    fea = []
    fea_bp = []
    fea_cc = []
    fea_mf = []
    combined_loader = zip(test_data, test_edge, test_LO, test_BP, test_CC, test_MF)
    with torch.no_grad():
        for i, (data_res, data_edge, data_lo, data_bp, data_cc, data_mf) in enumerate(combined_loader):
            data_res.cuda()
            data_edge.cuda()
            data_bp.cuda()
            data_cc.cuda()
            data_lo.cuda()
            data_mf.cuda()
            BP_out, CC_out, MF_out, BP_AH, CC_AH, MF_AH, LO_output, fea_st, feature_bp,feature_cc,feature_mf,Funrep,fea_GO,res_fea,Fea_BP, Fea_CC, Fea_MF = model(data_res, data_edge)

            fea_bp.append(Fea_BP)
            fea_cc.append(Fea_CC)
            fea_mf.append(Fea_MF)


            fea.append(fea_st)
            pre_lop = torch.cat((pre_lop, LO_output), dim=0)
            BP_out = BP_out.view(-1)
            CC_out = CC_out.view(-1)
            MF_out = MF_out.view(-1)
            LO_output = LO_output.view(-1)

            LO_predict = (LO_output > 0.5).float()
            LO_predict1 = LO_predict.view(-1, 10)
            pre_lo = torch.cat((pre_lo, LO_predict1), dim=0)
            lo_label = data_lo.y.view(-1, 10)
            true_lo = torch.cat((true_lo, lo_label), dim=0)

            BP_out_csv = BP_out.view(-1,119)
            pre_go_bp = torch.cat((pre_go_bp,BP_out_csv),dim=0)
            CC_out_csv = CC_out.view(-1,101)
            pre_go_cc = torch.cat((pre_go_cc,CC_out_csv),dim=0)
            MF_out_csv = MF_out.view(-1,69)
            pre_go_mf = torch.cat((pre_go_mf,MF_out_csv),dim=0)
            go_cc = data_cc.y.view(-1,101)
            true_go_cc = torch.cat((true_go_cc, go_cc),dim=0)
            go_bp = data_bp.y.view(-1, 119)
            true_go_bp = torch.cat((true_go_bp, go_bp), dim=0)
            go_mf = data_mf.y.view(-1, 69)
            true_go_mf = torch.cat((true_go_mf, go_mf), dim=0)


            y_pred_all_bp.append(BP_out)
            y_pred_all_cc.append(CC_out)
            y_pred_all_mf.append(MF_out)
            y_pred_all_lo.append(LO_output)

            real_all_bp.append(data_bp.y)
            real_all_cc.append(data_cc.y)
            real_all_mf.append(data_mf.y)
            real_all_lo.append(data_lo.y)
            lo_pre_label.append(LO_predict)

        y_pred_all_bp = torch.cat(y_pred_all_bp, dim=0).cpu()
        y_pred_all_cc = torch.cat(y_pred_all_cc, dim=0).cpu()
        y_pred_all_mf = torch.cat(y_pred_all_mf, dim=0).cpu()
        y_pred_all_lo = torch.cat(y_pred_all_lo, dim=0).cpu()
        lo_pre_label = torch.cat(lo_pre_label, dim=0).cpu()

        real_all_bp = torch.cat(real_all_bp, dim=0).cpu()
        real_all_cc = torch.cat(real_all_cc, dim=0).cpu()
        real_all_mf = torch.cat(real_all_mf, dim=0).cpu()
        real_all_lo = torch.cat(real_all_lo, dim=0).cpu()
        true_lo = true_lo.cpu()
        pre_lo = pre_lo.cpu()
        pre_lop = pre_lop.cpu()
        Hamming = hamming_loss(real_all_lo.numpy(), lo_pre_label.numpy())
        acc = accuracy1(real_all_lo.numpy(), lo_pre_label.numpy())
        abtr = absolute_true(true_lo.numpy(), pre_lo.numpy())
        coverage_rate = coverage_error(true_lo.numpy(), pre_lop.numpy())
        one_error_rate = one_error(true_lo.numpy(), pre_lop.numpy())
        Rloss = ranking_loss(true_lo.numpy(), pre_lop.numpy())
        class_precisions, aver_pre = calculate_class_precision(true_lo.numpy(), pre_lo.numpy())

        precision, recall, _ = precision_recall_curve(real_all_lo.numpy().flatten(), y_pred_all_lo.numpy().flatten())



        fpr_bp, tpr_bp, th_bp = roc_curve(real_all_bp.numpy().flatten(), y_pred_all_bp.numpy().flatten(), pos_label=1)
        auc_score_bp = auc(fpr_bp, tpr_bp)
        fpr_cc, tpr_cc, th_cc = roc_curve(real_all_cc.numpy().flatten(), y_pred_all_cc.numpy().flatten(), pos_label=1)
        auc_score_cc = auc(fpr_cc, tpr_cc)
        fpr_mf, tpr_mf, th_mf = roc_curve(real_all_mf.numpy().flatten(), y_pred_all_mf.numpy().flatten(), pos_label=1)
        auc_score_mf = auc(fpr_mf, tpr_mf)
        fpr_lo, tpr_lo, th_lo = roc_curve(real_all_lo.numpy().flatten(), y_pred_all_lo.numpy().flatten(), pos_label=1)
        auc_score_lo = auc(fpr_lo, tpr_lo)
        # Calculate AUPR for each label
        aupr_score_bp = cacul_aupr(real_all_bp.numpy().flatten(), y_pred_all_bp.numpy().flatten())
        aupr_score_cc = cacul_aupr(real_all_cc.numpy().flatten(), y_pred_all_cc.numpy().flatten())
        aupr_score_mf = cacul_aupr(real_all_mf.numpy().flatten(), y_pred_all_mf.numpy().flatten())
        aupr_score_lo = cacul_aupr(real_all_lo.numpy().flatten(), y_pred_all_lo.numpy().flatten())

        # Threshold predictions to binary values for F1, precision, and recall calculation
        each_best_fcore_bp = 0
        each_best_scores_bp = []
        for i in range(len(Thresholds)):
            f_score_bp, recall_bp, precision_bp = calculate_performance(
                real_all_bp.numpy(), y_pred_all_bp.numpy(), threshold=Thresholds[i], average='macro')
            if f_score_bp >= each_best_fcore_bp:
                each_best_fcore_bp = f_score_bp
                each_best_scores_bp = [Thresholds[i], f_score_bp, recall_bp, precision_bp, auc_score_bp, aupr_score_bp]

        each_best_fcore_bp1 = 0
        each_best_scores_bp1 = []
        for i in range(len(Thresholds)):
            f_score_bp1, recall_bp1, precision_bp1 = calculate_performance(
                real_all_bp.numpy(), y_pred_all_bp.numpy(), threshold=Thresholds[i], average='weighted')
            if f_score_bp1 >= each_best_fcore_bp1:
                each_best_fcore_bp1 = f_score_bp1
                each_best_scores_bp1 = [Thresholds[i], f_score_bp1, recall_bp1, precision_bp1, auc_score_bp,
                                        aupr_score_bp]

        each_best_fcore_cc = 0
        each_best_scores_cc = []
        for i in range(len(Thresholds)):
            f_score_cc, recall_cc, precision_cc = calculate_performance(
                real_all_cc.numpy(), y_pred_all_cc.numpy(), threshold=Thresholds[i], average='macro')
            if f_score_cc >= each_best_fcore_cc:
                each_best_fcore_cc = f_score_cc
                each_best_scores_cc = [Thresholds[i], f_score_cc, recall_cc, precision_cc, auc_score_cc, aupr_score_cc]

        each_best_fcore_cc1 = 0
        each_best_scores_cc1 = []
        for i in range(len(Thresholds)):
            f_score_cc1, recall_cc1, precision_cc1 = calculate_performance(
                real_all_cc.numpy(), y_pred_all_cc.numpy(), threshold=Thresholds[i], average='weighted')
            if f_score_cc1 >= each_best_fcore_cc1:
                each_best_fcore_cc1 = f_score_cc1
                each_best_scores_cc1 = [Thresholds[i], f_score_cc1, recall_cc1, precision_cc1, auc_score_cc,
                                        aupr_score_cc]

        each_best_fcore_mf = 0
        each_best_scores_mf = []
        for i in range(len(Thresholds)):
            f_score_mf, recall_mf, precision_mf = calculate_performance(
                real_all_mf.numpy(), y_pred_all_mf.numpy(), threshold=Thresholds[i], average='macro')
            if f_score_mf >= each_best_fcore_mf:
                each_best_fcore_mf = f_score_mf
                each_best_scores_mf = [Thresholds[i], f_score_mf, recall_mf, precision_mf, auc_score_mf, aupr_score_mf]

        each_best_fcore_mf1 = 0
        each_best_scores_mf1 = []
        for i in range(len(Thresholds)):
            f_score_mf1, recall_mf1, precision_mf1 = calculate_performance(
                real_all_mf.numpy(), y_pred_all_mf.numpy(), threshold=Thresholds[i], average='weighted')
            if f_score_mf1 >= each_best_fcore_mf1:
                each_best_fcore_mf1 = f_score_mf1
                each_best_scores_mf1 = [Thresholds[i], f_score_mf1, recall_mf1, precision_mf1, auc_score_mf,
                                        aupr_score_mf]

        each_best_fcore_lo1 = 0
        each_best_scores_lo1 = []
        for i in range(len(Thresholds)):
            f_score_lo1, recall_lo1, precision_lo1 = calculate_performance(
                real_all_lo.numpy(), y_pred_all_lo.numpy(), threshold=Thresholds[i], average='macro')
            if f_score_lo1 >= each_best_fcore_lo1:
                each_best_fcore_lo1 = f_score_lo1
                each_best_scores_lo1 = [Thresholds[i], f_score_lo1, recall_lo1, precision_lo1, auc_score_lo,
                                        aupr_score_lo]

        each_best_fcore_lo2 = 0
        each_best_scores_lo2 = []
        for i in range(len(Thresholds)):
            f_score_lo2, recall_lo2, precision_lo2 = calculate_performance(
                real_all_lo.numpy(), y_pred_all_lo.numpy(), threshold=Thresholds[i], average='weighted')
            if f_score_lo2 >= each_best_fcore_lo2:
                each_best_fcore_lo2 = f_score_lo2
                each_best_scores_lo2 = [Thresholds[i], f_score_lo2, recall_lo2, precision_lo2, auc_score_lo,
                                        aupr_score_lo]

        threshold = 0.5

        y_pred_all_bp_binary = (y_pred_all_bp > threshold).numpy().astype(int)
        y_pred_all_cc_binary = (y_pred_all_cc > threshold).numpy().astype(int)
        y_pred_all_mf_binary = (y_pred_all_mf > threshold).numpy().astype(int)
        y_pred_all_lo_binary = (y_pred_all_lo > threshold).numpy().astype(int)


        precision_bp_macro = precision_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='macro',
                                             zero_division=1.0)
        precision_cc_macro = precision_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='macro',
                                             zero_division=1.0)
        precision_mf_macro = precision_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='macro',
                                             zero_division=1.0)
        precision_lo_macro = precision_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='macro',
                                             zero_division=1.0)

        precision_bp_w = precision_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='weighted',
                                             zero_division=1.0)
        precision_cc_w = precision_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='weighted',
                                             zero_division=1.0)
        precision_mf_w = precision_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='weighted',
                                             zero_division=1.0)
        precision_lo_w = precision_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='weighted',
                                             zero_division=1.0)

        precision_bp_micro = precision_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='micro',
                                             zero_division=1.0)
        precision_cc_micro = precision_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='micro',
                                             zero_division=1.0)
        precision_mf_micro = precision_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='micro',
                                             zero_division=1.0)
        precision_lo_micro = precision_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='micro',
                                             zero_division=1.0)

        # Calculate recall for each label
        recall_bp_macro = recall_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='macro')
        recall_cc_macro = recall_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='macro')
        recall_mf_macro = recall_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='macro')
        recall_lo_macro = recall_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='macro')

        recall_bp_micro = recall_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='micro')
        recall_cc_micro = recall_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='micro')
        recall_mf_micro = recall_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='micro')
        recall_lo_micro = recall_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='micro')

        recall_bp_w = recall_score(real_all_bp.numpy(), y_pred_all_bp_binary, average='weighted')
        recall_cc_w = recall_score(real_all_cc.numpy(), y_pred_all_cc_binary, average='weighted')
        recall_mf_w = recall_score(real_all_mf.numpy(), y_pred_all_mf_binary, average='weighted')
        recall_lo_w = recall_score(real_all_lo.numpy(), y_pred_all_lo_binary, average='weighted')

        f_max_bp_mi = f_max(precision_bp_micro, recall_bp_micro)
        f_max_cc_mi = f_max(precision_cc_micro, recall_cc_micro)
        f_max_mf_mi = f_max(precision_mf_micro, recall_mf_micro)
        f_max_lo_mi = f_max(precision_lo_micro, recall_lo_micro)

        f_max_bp_w = f_max(precision_bp_w, recall_bp_w)
        f_max_cc_w = f_max(precision_cc_w, recall_cc_w)
        f_max_mf_w = f_max(precision_mf_w, recall_mf_w)
        f_max_lo_w = f_max(precision_lo_w, recall_lo_w)


        f_max_bp_ma = f_max(precision_bp_macro, recall_bp_macro)
        f_max_cc_ma = f_max(precision_cc_macro, recall_cc_macro)
        f_max_mf_ma = f_max(precision_mf_macro, recall_mf_macro)
        f_max_lo_ma = f_max(precision_lo_macro, recall_lo_macro)

        print("-------------test dataset-----------------")
        print("Fmax-mi - BP: {:.4f}, CC: {:.4f}, MF: {:.4f}, LO: {:.4f}".format(f_max_bp_mi, f_max_cc_mi,
                                                                             f_max_mf_mi, f_max_lo_mi))
        print("Fmax-ma - BP: {:.4f}, CC: {:.4f}, MF: {:.4f},  LO: {:.4f}".format(f_max_bp_ma, f_max_cc_ma,
                                                                 f_max_mf_ma, f_max_lo_ma))
        print("Fmax-w - BP: {:.4f}, CC: {:.4f}, MF: {:.4f},  LO: {:.4f}".format(f_max_bp_w, f_max_cc_w,
                                                                 f_max_mf_w, f_max_lo_w))

        print('BP-macro:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_bp[4],
                                                                                             each_best_scores_bp[5],
                                                                                             each_best_scores_bp[2],
                                                                                             each_best_scores_bp[3],
                                                                                             each_best_scores_bp[1]))
        print('BP-weighted:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_bp1[4],
                                                                                                each_best_scores_bp1[5],
                                                                                                each_best_scores_bp1[2],
                                                                                                each_best_scores_bp1[3],
                                                                                                each_best_scores_bp1[
                                                                                                    1]))
        print('CC-macro:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_cc[4],
                                                                                             each_best_scores_cc[5],
                                                                                             each_best_scores_cc[2],
                                                                                             each_best_scores_cc[3],
                                                                                             each_best_scores_cc[1]))
        print('CC-weighted:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_cc1[4],
                                                                                                each_best_scores_cc1[5],
                                                                                                each_best_scores_cc1[2],
                                                                                                each_best_scores_cc1[3],
                                                                                                each_best_scores_cc1[
                                                                                                    1]))
        print('MF-macro:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_mf[4],
                                                                                             each_best_scores_mf[5],
                                                                                             each_best_scores_mf[2],
                                                                                             each_best_scores_mf[3],
                                                                                             each_best_scores_mf[1]))
        print('MF-weighted:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_mf1[4],
                                                                                                each_best_scores_mf1[5],
                                                                                                each_best_scores_mf1[2],
                                                                                                each_best_scores_mf1[3],
                                                                                                each_best_scores_mf1[
                                                                                                    1]))

        print('LO-macro:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_lo1[4],
                                                                                               each_best_scores_lo1[5],
                                                                                               each_best_scores_lo1[2],
                                                                                               each_best_scores_lo1[3],
                                                                                               each_best_scores_lo1[1]))
        print(
            'LO-weighted:---  AUC:{}   AUPR:{}    Recall:{}     Precision:{}    F1:{}'.format(each_best_scores_lo2[4],
                                                                                                each_best_scores_lo2[5],
                                                                                                each_best_scores_lo2[2],
                                                                                                each_best_scores_lo2[3],
                                                                                                each_best_scores_lo2[
                                                                                                    1]))
        print('LO：--Hamming: {}，  Accuracy:{}, CV:{},  AT:{},  RL:{}, Onero:{}'.format(Hamming,acc, coverage_rate,abtr,Rloss, one_error_rate))
        print('CPR:{}'.format(class_precisions))
        print('AP:{}'.format(aver_pre))



if __name__=="__main__" :
    start_time = time.time()
    set_seed()
    train()
    end_time = time.time()
    final_time = (end_time - start_time) / 60
    print(f"train  took {final_time:.2f} minutes")




