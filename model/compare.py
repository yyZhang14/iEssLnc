import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def getData():
    xiao = pd.read_excel("../data/compare/Mouse.xlsx")
    print("xiao len:",len(xiao))
    test = list(xiao["Name"])
    dataset = pd.read_csv("../result/mouse/para/300_40_32/dataset2.csv")
    print("all dataset:",len(dataset))
    testData = dataset[dataset['0'].isin(test)]
    trainData = dataset[~dataset['0'].isin(test)]

    x_data = trainData.iloc[:, 1:33]
    x_data = np.array(x_data)
    x_label = trainData.iloc[:, -1]
    x_label = np.array(x_label)

    y_data = testData.iloc[:, 1:33]
    y_data = np.array(y_data)
    y_label = testData.iloc[:, -1]
    y_label = np.array(y_label)

    return x_data,x_label,y_data,y_label

def plot_figure():
    y_h = [71.43, 91.80, 75.41]
    x_h = ['SGII_human', 'PredEssLnc_H1', 'PredEssLnc_H2']
    y_m = [87.50, 100.00, 85.71]
    x_m = ['SGII_mouse', 'PredEssLnc_M1','PredEssLnc_M2']

    plt.bar(x_h, y_h, color="#FCE2DB", width = 0.5, label='Human dataset')
    plt.bar(x_m, y_m, color="#FF8FB1", width = 0.5, label='Mouse dataset')
    plt.ylabel('Score(%)')
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    for a,b in zip(x_h,y_h):   #柱子上的数字显示
        plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
    for a, b in zip(x_m, y_m):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7);
    #plt.show()
    plt.savefig('../img/sen.png', bbox_inches='tight')


if __name__ == '__main__':
    #plot_figure()
    train_data,train_label,test_data,test_label = getData()
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    print(test_label)
    clf = SVC(C=10, kernel='rbf', gamma=0.01)
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    print(pred_label)

    TN, FP, FN, TP = confusion_matrix(test_label, pred_label).ravel()
    print(TP, TN, FP, FN)
    accuracy = accuracy_score(test_label, pred_label)
    precision = precision_score(test_label, pred_label)
    sensitivity = recall_score(test_label, pred_label)


    print("accuracy:  %0.4f"% accuracy)
    print("precision:  %0.4f" % precision)
    print("sensitivity/recall:  %0.4f" % sensitivity)
