import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef,precision_recall_curve, auc


def get_data():
    # 读取数据集
    res = pd.read_csv("../result/human/para/300_30_128/dataset2.csv", header=0, index_col=0)
    length = len(res)
    result = res.iloc[:, 0:128]
    result = np.array(result)
    flag = res.iloc[:, -1]
    flag = np.array(flag)
    return result, flag, length

if __name__ == '__main__':
    data, label, set_len = get_data()
    loo = LeaveOneOut()

    predict_label_list = []
    real_label_list = []
    count = 0 # 循环次数
    y_score_list = []

    rfc = RandomForestClassifier(max_depth=9, n_estimators=50, random_state=0)
    for train_index, test_index in loo.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        # scaler = StandardScaler()
        # scaler = scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        rfc.fit(x_train, y_train)
        pred_temp = rfc.predict(x_test)

        predict_label_list.append(list(pred_temp))
        real_label_list.append(list(y_test))

        #predicy_proda 概率
        y_score_temp = rfc.predict_proba(x_test)
        y_score_list.append(y_score_temp[:, 1])

        count += 1
        print("第{}次循环".format(count))

    print("循环结束")
    real_label_list = np.array(real_label_list).flatten()
    predict_label_list = np.array(predict_label_list).flatten()
    # print(real_label_list)
    # print(predict_label_list)
    TN, FP, FN, TP = confusion_matrix(real_label_list, predict_label_list).ravel()

    print(TP, TN, FP, FN)
    MCC1 = matthews_corrcoef(real_label_list, predict_label_list)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    sensitivity = TP/(TP+FN)
    specificity = TN / (TN + FP)
    F1_score = (2* precision * sensitivity) / (precision + sensitivity)

    print("accuracy:  %0.3f"% accuracy)
    print("precision:  %0.3f" % precision)
    print("sensitivity/recall:  %0.3f" % sensitivity)
    print("specificity:  %0.3f" % specificity)
    print("F1_score:  %0.3f" % F1_score)
    print("MCC1 : %0.3f" %MCC1)

    pred_score = np.array(y_score_list).flatten()
    fpr, tpr, threshold = metrics.roc_curve(real_label_list, pred_score)
    roc_auc = metrics.auc(fpr, tpr)
    print("AUROC: %.3f"% roc_auc)

    pre, rec, thresholds = precision_recall_curve(real_label_list, pred_score)
    aupr = auc(rec,pre)
    print("AUPR:%0.3f" % aupr)

    # print("fpr", fpr)
    # print("tpr", tpr)


    # f = open(r"C:\Users\zyy\Desktop\code\prediction esslnc\result\GIC\human\RF.txt","a+")
    # real_temp = ""
    # score_temp=""
    # for i in real_label_list:
    #     real_temp += str(i)+","
    # for j in pred_score:
    #     score_temp += str(j)+","
    # f.write(real_temp+"\n")
    # f.write(score_temp+"\n")
    # f.close()



    plt.figure()
    plt.plot(fpr, tpr, color='#53BF9D', lw=1, label='RF AUROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Human dataset1')
    plt.legend(loc="lower right")
    plt.show()