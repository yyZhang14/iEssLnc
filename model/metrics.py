import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, \
    precision_score, recall_score, f1_score

def get_performance(real_label_list,predict_label_list,y_score_list):
    real_label_list = np.array(real_label_list).flatten()
    predict_label_list = np.array(predict_label_list).flatten()
    pred_score = np.array(y_score_list).flatten()

    TN, FP, FN, TP = confusion_matrix(real_label_list, predict_label_list).ravel()
    MCC = matthews_corrcoef(real_label_list, predict_label_list)
    accuracy = accuracy_score(real_label_list, predict_label_list)
    precision = precision_score(real_label_list, predict_label_list)
    sensitivity = recall_score(real_label_list, predict_label_list)
    specificity = TN / (TN + FP)
    F1_score = f1_score(real_label_list, predict_label_list)
    fpr, tpr, threshold = metrics.roc_curve(real_label_list, pred_score)
    roc_auc = metrics.auc(fpr, tpr)
    pre, rec, thresholds = precision_recall_curve(real_label_list, pred_score)
    aupr = auc(rec, pre)
    res = dict()
    res["accuracy"] = round(accuracy, 3)
    res["precision"] = round(precision, 3)
    res["sensitivity"] = round(sensitivity, 3)
    res["specificity"] = round(specificity, 3)
    res["F1_score"] = round(F1_score, 3)
    res["MCC"] = round(MCC, 3)
    res["roc_auc"] = round(roc_auc, 3)
    res["aupr"] = round(aupr, 3)
    return res,pre,rec