import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, \
    precision_score, recall_score, f1_score


def get_data():
    # 读取数据集
    res = pd.read_csv("../result/human/para/300_30_128/dataset2.csv", header=0,index_col=0)

    print(len(res))
    result = res.iloc[:, 0:128]
    result = np.array(result)

    flag = res.iloc[:, -1]
    flag = np.array(flag)
    print(result.shape)
    print(flag.shape)
    return result, flag

if __name__ == '__main__':
    data, label = get_data()

    loo = LeaveOneOut()

    predict_label_list = []
    real_label_list = []

    count = 0 # 循环次数
    y_score_list = []

    clf = SVC(C=10, kernel='rbf',gamma=0.01)

    for train_index, test_index in loo.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        clf.fit(x_train, y_train)
        pred_temp = clf.predict(x_test)

        predict_label_list.append(list(pred_temp))
        real_label_list.append(list(y_test))

        # 使用decision_function 得到测试集的分数
        y_score_temp = clf.decision_function(x_test)
        y_score_list.append(y_score_temp)

        count += 1
        print("第{}次循环".format(count))

    print("循环结束")

    real_label_list = np.array(real_label_list).flatten()
    predict_label_list = np.array(predict_label_list).flatten()

    TN, FP, FN, TP = confusion_matrix(real_label_list, predict_label_list).ravel()
    print(TP, TN, FP, FN)
    MCC = matthews_corrcoef(real_label_list, predict_label_list)
    accuracy = accuracy_score(real_label_list, predict_label_list)
    precision = precision_score(real_label_list, predict_label_list)
    sensitivity = recall_score(real_label_list, predict_label_list)
    specificity = TN / (TN + FP)
    F1_score = f1_score(real_label_list, predict_label_list)

    print("accuracy:  %0.3f"% accuracy)
    print("precision:  %0.3f" % precision)
    print("sensitivity/recall:  %0.3f" % sensitivity)
    print("specificity:  %0.3f" % specificity)
    print("F1_score:  %0.3f" % F1_score)
    print("MCC : %0.3f" % MCC)

    pred_score = np.array(y_score_list).flatten()
    fpr, tpr, threshold = metrics.roc_curve(real_label_list, pred_score)
    roc_auc = metrics.auc(fpr, tpr)
    print("AUROC: %.3f" % roc_auc)

    pre, rec, thresholds = precision_recall_curve(real_label_list, pred_score)
    aupr = auc(rec, pre)
    print("AUPR: %0.3f" % aupr)

    # f = open(r"C:\Users\zyy\Desktop\code\prediction esslnc\result\mouse\SVM.txt","a+")
    # real_temp = ""
    # score_temp=""
    # for i in real_label_list:
    #     real_temp += str(i)+","
    # for j in pred_score:
    #     score_temp += str(j)+","
    # f.write(real_temp+"\n")
    # f.write(score_temp+"\n")
    # f.close()

    #
    # print("fpr", fpr)
    # print("tpr", tpr)
    #
    plt.figure()
    plt.plot(fpr, tpr, color='#53BF9D', lw=1, label='SVM AUROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Human dataset1')
    plt.legend(loc="lower right")
    plt.show()

