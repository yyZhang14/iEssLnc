import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, \
    precision_score, recall_score, f1_score

def get_data(filename):
    data = dict()
    name = ["real1", "pred1", "real2", "pred2"]
    f = open(filename, "r")
    i = 0
    for line in f.readlines():
        line = line.strip("\n").strip(",").split(",")
        numbers = [float(x) for x in line]
        data[name[i]] = numbers
        i += 1
    f.close()
    return data


def plot_figure(real_RF, pred_RF, real_SVM, pred_SVM, real_CNN, pred_CNN, text):
    fpr_RF, tpr_RF, threshold = roc_curve(real_RF, pred_RF)
    roc_auc_RF = auc(fpr_RF, tpr_RF)
    print("roc_auc_RF:",roc_auc_RF)

    fpr_SVM, tpr_SVM, threshold = roc_curve(real_SVM, pred_SVM)
    roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
    print("roc_auc_SVM",roc_auc_SVM)

    fpr_CNN, tpr_CNN, threshold = roc_curve(real_CNN, pred_CNN)
    roc_auc_CNN = auc(fpr_CNN, tpr_CNN)
    print("roc_auc_CNN",roc_auc_CNN)

    plt.figure()
    plt.plot(fpr_RF, tpr_RF, color='#97DBAE', lw=2, label='RF AUROC (area = %0.3f)' % roc_auc_RF)
    plt.plot(fpr_SVM, tpr_SVM, color='#F4BBBB', lw=2, label='SVM AUROC (area = %0.3f)' % roc_auc_SVM)
    plt.plot(fpr_CNN, tpr_CNN, color='#FBB454', lw=2, label='CNN AUROC (area = %0.3f)' % roc_auc_CNN)

    # plt.plot([0, 1], [0, 1], color='dimgrey', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(text)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(r'C:\Users\zyy\Desktop\code\prediction esslnc\result\mouse\mouse_dataset2.png', bbox_inches='tight')


if __name__ == '__main__':
    dirpath = r'C:\Users\zyy\Desktop\code\prediction esslnc\result'

    data_RF = get_data(dirpath + '/' + 'mouse' + '/' + 'RF' + '.txt')
    data_SVM = get_data(dirpath + '/' + 'mouse' + '/' + 'SVM' + '.txt')
    data_CNN = get_data(dirpath + '/' + 'mouse' + '/' + 'CNN' + '.txt')

    #plot_figure(data_RF['real1'], data_RF['pred1'], data_SVM['real1'], data_SVM['pred1'], data_CNN['real1'], data_CNN['pred1'],"M1 dataset")
    plot_figure(data_RF['real2'], data_RF['pred2'], data_SVM['real2'], data_SVM['pred2'], data_CNN['real2'],data_CNN['pred2'], "M2 dataset")
