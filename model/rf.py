import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from metrics import get_performance


path = "../result/human/para/300_30_256/dataset1.csv"

def get_data():
    res = pd.read_csv(path, header=0, index_col=0)

    length = len(res)
    result = res.iloc[:, 0:256]
    result = np.array(result)
    flag = res.iloc[:, -1]
    flag = np.array(flag)
    return result, flag, length

if __name__ == '__main__':
    data, label, set_len = get_data()
    loo = LeaveOneOut()
    predict_label_list = []
    real_label_list = []
    count = 0
    y_score_list = []

    rfc = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
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
    res, pre, rec = get_performance(real_label_list, predict_label_list, y_score_list)
    print(res)



