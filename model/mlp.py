from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from metrics import get_performance

path = "../result/human/para/300_30_256/dataset1.csv"

def get_data():
    # 读取数据集
    res = pd.read_csv(path, header=0,index_col=0)
    print(len(res))
    result = res.iloc[:, 0:256]
    result = np.array(result)

    flag = res.iloc[:, -1]
    flag = np.array(flag)
    print(result.shape)
    print(flag.shape)
    return result, flag

if __name__ == '__main__':
    data, label = get_data()
    loo = LeaveOneOut()
    layers = [32,64,128,256]
    max_iters = [200,400,600,800,1000]
    f=open("../result/human/MLP_para.csv","w",encoding="utf-8")
    f.write("layer"+","+"max_iter"+","+"accuracy"+","+"precision"+","+"sensitivity"
            +","+"specificity"+","+"F1_score"+","+"MCC"+","+"AUROC"+","+"AUPR"+"\n")
    f.close()
    for i in layers:
        for j in max_iters:
            count = 0
            predict_label_list = []
            real_label_list = []
            y_score_list = []
            clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                            hidden_layer_sizes=(i,i), random_state=1, max_iter=j, verbose=True)

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
            data,pre,rec = get_performance(real_label_list, predict_label_list, y_score_list)

            a = str(i) + "," + str(j)+","+str(data["accuracy"]) + "," + str(data["precision"]) + "," + str(data["sensitivity"]) + "," + \
                str(data["specificity"]) + "," + str(data["F1_score"]) + "," + str(data["MCC"]) + "," + str(data["roc_auc"]) + "," + str(data["aupr"]) + "\n"

            f = open("../result/human/MLP_para.csv", "a+", encoding="utf-8")
            f.write(a)
            f.close()


