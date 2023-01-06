import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from metrics import get_performance

def get_data(path,r,index):
    # read dataset
    if isinstance(path, pd.DataFrame):
        res = path
    else:
        res = pd.read_csv(path, header=0, index_col=0)
    result = res.iloc[:, 0:r]
    result = np.array(result)

    flag = res.iloc[:, index]
    flag = np.array(flag)
    print(result.shape)
    print(flag.shape)
    return result, flag


def svm_model(path,r,gamma,C,index):
    data, label = get_data(path,r,index)
    loo = LeaveOneOut()
    clf = SVC(kernel="rbf", gamma=gamma, C=C)
    predict_label_list = []
    real_label_list = []
    count = 0
    y_score_list = []

    for train_index, test_index in loo.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        mms = MinMaxScaler()
        x_train = mms.fit_transform(x_train)
        x_test = mms.transform(x_test)

        clf.fit(x_train, y_train)
        pred_temp = clf.predict(x_test)

        predict_label_list.append(list(pred_temp))
        real_label_list.append(list(y_test))

        # use decision_function to get score
        y_score_temp = clf.decision_function(x_test)
        y_score_list.append(y_score_temp)

        count += 1
    print("Run round {}".format(count))
    print("End of program round!")
    data,pre,rec = get_performance(real_label_list, predict_label_list, y_score_list)

    return data,predict_label_list,pre,rec

