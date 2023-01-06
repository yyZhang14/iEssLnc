from svm_module import svm_model
import pandas as pd

if __name__ == '__main__':

    # dataset1 train :get true label and pred label
    # path = "../result/human/para/300_30_256/dataset1.csv"
    # r = 256
    # gamma = 0.1
    # C = 1
    # index = -1
    # data,pred_label ,precision,recall= svm_model(path, r, gamma, C,index)
    # f = open("../result/ROCPR/human/SVM.txt","a+")
    # for i in precision:
    #     f.write(str(i)+',')
    # f.write("\n")
    # for i in recall:
    #     f.write(str(i)+',')
    # f.close()


    # shuffle
    path = "../result/shuffle/400_100_64/dataset1.csv"
    r = 64
    gamma = 0.001
    C = 100
    f = open("../result/shuffle/400_100_64/mouse_shuffle_SVM.csv", "a+")
    f.write("label"+','+"accuracy"+','+"precision"+','+"sensitivity"+','+"specificity"+','+"F1_score"+','+"MCC"+','+"AUROC"+','+"AUPR"+'\n')
    for index in range(-1001,0):
        print(index)
        data, pred_label, precision, recall = svm_model(path, r, gamma, C,index)

        a = str(index)+','+str(data["accuracy"])+","+str(data["precision"])+","+str(data["sensitivity"])+","+\
            str(data["specificity"])+","+str(data["F1_score"])+","+str(data["MCC"])+","+str(data["roc_auc"])+","+str(data["aupr"])+"\n"
        f.write(a)
    f.close()




