import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import  precision_recall_curve

def get_data(filename):
    data = dict()
    name = ["real", "pred"]
    f = open(filename, "r")
    i = 0
    for line in f.readlines():
        line = line.strip("\n").strip(",").split(",")
        numbers = [float(x) for x in line]
        data[name[i]] = numbers
        i += 1
    f.close()
    return data


def plot_ROCfigure(real_RF, pred_RF, real_SVM, pred_SVM, real_MLP, pred_MLP, text):
    fpr_RF, tpr_RF, threshold = roc_curve(real_RF, pred_RF)
    roc_auc_RF = auc(fpr_RF, tpr_RF)
    print("roc_auc_RF:",roc_auc_RF)

    fpr_SVM, tpr_SVM, threshold = roc_curve(real_SVM, pred_SVM)
    roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
    print("roc_auc_SVM",roc_auc_SVM)

    fpr_MLP, tpr_MLP, threshold = roc_curve(real_MLP, pred_MLP)
    roc_auc_MLP = auc(fpr_MLP, tpr_MLP)
    print("roc_auc_MLP",roc_auc_MLP)

    plt.figure()
    plt.plot(fpr_RF, tpr_RF, color='#97DBAE', lw=2, label='RF (AUROC = %0.3f)' % roc_auc_RF)
    plt.plot(fpr_SVM, tpr_SVM, color='#F4BBBB', lw=2, label='SVM (AUROC = %0.3f)' % roc_auc_SVM)
    plt.plot(fpr_MLP, tpr_MLP, color='#FBB454', lw=2, label='MLP (AUROC = %0.3f)' % roc_auc_MLP)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(text)
    plt.legend(loc="lower right")
    plt.savefig('../result/ROCPR/human/ROC_human.svg', bbox_inches='tight')
    plt.show()


def plot_PRfigure(real_RF, pred_RF, real_SVM, pred_SVM, real_MLP, pred_MLP, text):

    pre_RF, rec_RF, thresholds = precision_recall_curve(real_RF, pred_RF)
    aupr_RF = auc(rec_RF,pre_RF)
    print("AUPR_RF:",aupr_RF)

    pre_SVM, rec_SVM, thresholds = precision_recall_curve(real_SVM, pred_SVM)
    aupr_SVM = auc(rec_SVM,pre_SVM)
    print("AUPR_SVM:",aupr_SVM)

    pre_MLP, rec_MLP, thresholds = precision_recall_curve(real_MLP, pred_MLP)
    aupr_MLP = auc(rec_MLP,pre_MLP)
    print("AUPR_MLP:",aupr_MLP)

    plt.figure()
    plt.plot(rec_RF,pre_RF, color='#97DBAE', lw=2, label='RF (AUPR = %0.3f)' % aupr_RF)
    plt.plot(rec_SVM,pre_SVM, color='#F4BBBB', lw=2, label='SVM (AUPR = %0.3f)' % aupr_SVM)
    plt.plot(rec_MLP,pre_MLP, color='#FBB454', lw=2, label='MLP (AUPR = %0.3f)' % aupr_MLP)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(text)
    plt.legend(loc="lower right")

    plt.savefig('../result/ROCPR/mouse/PR_mouse.svg', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    dirpath = '../result/ROCPR'
    data_RF = get_data(dirpath + '/' + 'human' + '/' + 'RF' + '.txt')
    data_SVM = get_data(dirpath + '/' + 'human' + '/' + 'SVM' + '.txt')
    data_MLP = get_data(dirpath + '/' + 'human' + '/' + 'MLP' + '.txt')
    #
    plot_ROCfigure(data_RF['real'], data_RF['pred'], data_SVM['real'], data_SVM['pred'],
                data_MLP['real'], data_MLP['pred'],"Human(ROC)")
    plot_PRfigure(data_RF['real'], data_RF['pred'], data_SVM['real'],
                        data_SVM['pred'], data_MLP['real'],data_MLP['pred'], "Human(PR)")



