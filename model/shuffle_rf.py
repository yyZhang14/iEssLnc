import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from metrics import get_performance


def get_data(index):
    # read dataset
    res = pd.read_csv("../result/shuffle/400_100_64/dataset1.csv", header=0, index_col=0)
    print(len(res))
    result = res.iloc[:, 0:64]
    result = np.array(result)

    flag = res.iloc[:, index]
    flag = np.array(flag)
    print(result.shape)
    print(flag.shape)
    return result, flag

if __name__ == '__main__':

    f = open("../result/shuffle/400_100_64/mouse_shuffle_RF.csv", "a+")
    f.write("label"+','+"accuracy"+','+"precision"+','+"sensitivity"+','+"specificity"+','+"F1_score"+','+"MCC"+','+"AUROC"+','+"AUPR"+'\n')
    for index in range(-1001,0):
        print(index)
        data, label = get_data(index)
        loo = LeaveOneOut()
        rfc = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=0)

        predict_label_list = []
        real_label_list = []
        count = 0
        y_score_list = []

        for train_index, test_index in loo.split(data):
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            rfc.fit(x_train, y_train)
            pred_temp = rfc.predict(x_test)

            predict_label_list.append(list(pred_temp))
            real_label_list.append(list(y_test))

            #predic_proda
            y_score_temp = rfc.predict_proba(x_test)
            y_score_list.append(y_score_temp[:, 1])

            count += 1
        print("Run round {}".format(count))
        print("End of program round!")
        result, pre, rec = get_performance(real_label_list, predict_label_list, y_score_list)

        a = "label"+str(1000+index) + ',' + str(result["accuracy"]) + "," + str(result["precision"]) + "," + str(result["sensitivity"]) + "," + \
            str(result["specificity"]) + "," + str(result["F1_score"]) + "," + str(result["MCC"]) + "," + str(result["roc_auc"]) + "," + str(result["aupr"]) + "\n"
        f.write(a)
    f.close()


