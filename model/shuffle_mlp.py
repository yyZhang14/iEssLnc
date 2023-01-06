from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from metrics import get_performance

def get_data(index):
    res = pd.read_csv("../result/shuffle/300_30_256/dataset1.csv", header=0, index_col=0)
    result = res.iloc[:, 0:256]
    result = np.array(result)

    flag = res.iloc[:, index]
    flag = np.array(flag)
    return result, flag

if __name__ == '__main__':

    f = open("../result/shuffle/300_30_256/human_shuffle_MLP.csv", "a+")
    f.write("label"+','+"accuracy"+','+"precision"+','+"sensitivity"+','+"specificity"+','+"F1_score"+','+"MCC"+','+"AUROC"+','+"AUPR"+'\n')
    for index in range(-1001,0):
        print(index)
        data, label = get_data(index)
        loo = LeaveOneOut()
        count = 0

        predict_label_list = []
        real_label_list = []
        y_score_list = []
        clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                        hidden_layer_sizes=(64,64), random_state=1, max_iter=200, verbose=True)

        for train_index, test_index in loo.split(data):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            mms = MinMaxScaler()
            x_train = mms.fit_transform(x_train)
            x_test = mms.transform(x_test)

            clf.fit(x_train, y_train)
            pred_temp = clf.predict(x_test)

            predict_label_list.append(pred_temp)
            real_label_list.append(y_test)

            # 使用decision_function 得到测试集的分数
            y_score_temp = clf.predict_proba(x_test)
            y_score_list.append(y_score_temp[:,1])

            count += 1
        print("Run round {}".format(count))
        print("End of program round!")
        result, pre, rec = get_performance(real_label_list, predict_label_list, y_score_list)
        a = "label" + str(1000 + index) + ',' + str(result["accuracy"]) + "," + str(result["precision"]) + "," + \
            str(result["sensitivity"]) + "," + str(result["specificity"]) + "," + str(result["F1_score"]) + "," + str(result["MCC"]) \
            + "," + str(result["roc_auc"]) + "," + str(result["aupr"]) + "\n"
        f.write(a)
    f.close()



