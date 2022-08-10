import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv1D,Dense, Activation,Dropout,MaxPooling1D
from keras import optimizers
from keras import regularizers

from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, \
    precision_score, recall_score, f1_score, roc_curve
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data():
    # 读取数据集
    res = pd.read_csv("../result/mouse/para/300_40_32/dataset1.csv", header=0,
                      index_col=0)
    print(len(res))
    result = res.iloc[:, 0:32]
    result = np.array(result)

    flag = res.iloc[:, -1]
    flag = np.array(flag)
    return result, flag


def get_model():
    model = Sequential()
    model.add(Conv1D(filters = 128,kernel_size=3, strides=1,padding='valid',activation='relu',input_shape=(128,1)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)
                    ))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_para(true_label, pred_label):
    TN, FP, FN, TP = confusion_matrix(true_label, pred_label).ravel()
    print(TP, TN, FP, FN)
    MCC = matthews_corrcoef(true_label, pred_label)
    accuracy = accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    sensitivity = recall_score(true_label, pred_label)
    specificity = TN / (TN + FP)
    F1_score = f1_score(true_label, pred_label)

    print("accuracy:  %0.3f" % accuracy)
    print("precision:  %0.3f" % precision)
    print("sensitivity:  %0.3f" % sensitivity)
    print("specificity:  %0.3f" % specificity)
    print("F1_score:  %0.3f" % F1_score)
    print("MCC : %0.3f" % MCC)


def get_figure(true_label, pred_label):
    fpr, tpr, threshold = roc_curve(true_label, pred_label)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color='#97DBAE', lw=2, label='CNN AUROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='dimgrey', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("mouse dataset1")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    data, label = get_data()
    loo = LeaveOneOut()

    predict_label_list = []
    real_label_list = []

    count = 0 # 循环次数
    y_score_list = []

    cnn = get_model()
    cnn.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.01),
                metrics=['binary_accuracy'])

    for train_index, test_index in loo.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        x_train = x_train.reshape(x_train.shape[0], 128, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 128, 1).astype('float32')

        H = cnn.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0)

        pred_temp = cnn.predict(x_test)

        predict_label_list.append(list(pred_temp))
        real_label_list.append(list(y_test))

        count += 1
        print("第{}次循环".format(count))
    print("循环结束")

    real_label_list = np.array(real_label_list).flatten()
    predict_label_list = np.array(predict_label_list).flatten()
    print(real_label_list)
    print(predict_label_list)

    # get_para(real_label_list, predict_label_list)
    # get_figure(real_label_list, predict_label_list)

    f = open(r"C:\Users\zyy\Desktop\data\TPR_FPR\mouse\CNN.txt","a+")
    real_temp = ""
    score_temp=""
    for i in real_label_list:
        real_temp += str(i)+","
    for j in predict_label_list:
        score_temp += str(j)+","
    f.write(real_temp+"\n")
    f.write(score_temp+"\n")
    f.close()