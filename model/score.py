import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


def get_data():
    res = pd.read_csv("../result/mouse/para/400_100_64/dataset1.csv", header=0, index_col=0)
    print(len(res))
    result = res.iloc[:, 0:64]
    result = np.array(result)

    flag = res.iloc[:, -1]
    flag = np.array(flag)
    print(result.shape)
    print(flag.shape)

    alldata = pd.read_csv("../result/mouse/para/400_100_64/alldata.csv", header=0, index_col=0)
    alldata = alldata.iloc[:, 0:64]
    name = alldata.index.values
    alldata = np.array(alldata)

    return result, flag, alldata, name


if __name__ == '__main__':

    x_train, y_train, alldata, name = get_data()
    print(len(name))
    # MLP model
    clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                        hidden_layer_sizes=(4, 4), random_state=1, max_iter=200, verbose=True)

    mms = MinMaxScaler()
    x_train = mms.fit_transform(x_train)
    alldata = mms.transform(alldata)

    clf.fit(x_train, y_train)
    pred = clf.predict(alldata)

    y_score_temp = clf.predict_proba(alldata)
    y_score_list = y_score_temp[:, 1]

    print(len(y_score_list))

    f = open("../result/MLP_score/scoreM.csv", "a+",encoding='utf-8')
    f.write("name" + ',' + "score" +","+ "pred_label"+'\n')
    for i in range(0, len(name)):
        ptr = str(name[i]) + ',' + str(y_score_list[i]) + ','+str(pred[i])+'\n'
        f.write(ptr)
    f.close()

    # SVM model
    # clf = SVC(kernel="rbf", gamma=0.001, C=100)
    #
    # mms = MinMaxScaler()
    # x_train = mms.fit_transform(x_train)
    # alldata = mms.transform(alldata)
    #
    # clf.fit(x_train, y_train)
    # pred = clf.predict(alldata)
    # y_score = clf.decision_function(alldata)
    #
    # f = open("../result/mouse/para/400_100_64/gene_predLabel.csv", "a+",encoding="utf-8")
    # f.write("name" + ','+"score"+"," + "pred label" + '\n')
    # for i in range(0, len(name)):
    #     ptr = str(name[i]) + ',' +str(y_score[i])+"," + str(pred[i]) + '\n'
    #     f.write(ptr)
    # f.close()

    # RF model
    # y_score_list = []
    # rfc = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=0)
    # rfc.fit(x_train, y_train)
    # pred = rfc.predict(alldata)
    #
    # y_score_temp = rfc.predict_proba(alldata)
    # y_score_list = y_score_temp[:, 1]
    #
    # f = open("../result/RF_score/scoreM.csv", "a+",encoding='utf-8')
    # f.write("name" + ',' + "score" +","+ "pred_label"+'\n')
    # for i in range(0, len(name)):
    #     ptr = str(name[i]) + ',' + str(y_score_list[i]) + ','+str(pred[i])+'\n'
    #     f.write(ptr)
    # f.close()

