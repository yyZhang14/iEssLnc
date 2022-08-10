import pandas as pd
import numpy as np
import random

def get_data():
    #Human
    pos = pd.read_csv("../data/mouse/essential.csv")
    positive = list(pos["lncRNA"])
    f1 = open("../result/mouse/para/300_40_32/out_mouse_300_40.txt", "r", encoding="utf-8")
    pos_data = []
    unlabel = []

    for line in f1.readlines():
        line = line.strip().split(" ")
        if line[0][0] == 'l':
            line[0] = line[0][1:]
            lnc = line[0]
            if lnc in positive:
                pos_data.append(line)
            else:
                unlabel.append(line)
    f1.close()
    pos_data = pd.DataFrame(pos_data)
    pos_data["label"] = 1
    unlabel_data = pd.DataFrame(unlabel)
    unlabel_data["label"] = 0
    U = list(unlabel_data[0])
    neg1 = random.sample(U, 25)
    neg2 = random.sample(U, 50)
    neg1_data = unlabel_data[unlabel_data[0].isin(neg1)]
    neg2_data = unlabel_data[unlabel_data[0].isin(neg2)]

    dataset1 = pd.concat([pos_data, neg1_data], axis=0)
    dataset2 = pd.concat([pos_data, neg2_data], axis=0)
    print("dataset1", len(dataset1))
    print("dataset2", len(dataset2))

    dataset1.to_csv('../data/compare/mouse_dataset1.csv', index=False)
    dataset2.to_csv('../data/compare/mouse_dataset2.csv', index=False)


if __name__ == '__main__':
    get_data()