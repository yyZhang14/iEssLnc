"""
1. 处理out_human_1000_100.txt 文件
2. 挑选出 postive 的向量
3. 分别挑选出negative 的向量
4. 生成数据集1 和数据集2
"""
import os
import pandas as pd
import numpy as np

class Pro_Data:
    def __init__(self):
        self.positive = []
        self.negative1 = []
        self.negative2 = []


    def get_pos_nag(self,filename):
        f = open(filename, "r", encoding="utf-8")
        res = []
        for line in f.readlines():
            line = line.strip().split(" ")
            res.append(line)
        f.close()
        res = np.array(res).flatten()
        print(len(res))
        return res

    def get_path(self, filename):
        f1 = open(filename,"r",encoding="utf-8")
        pos_data = []
        neg1_data = []
        neg2_data =[]

        for line in f1.readlines():
            line = line.strip().split(" ")
            if line[0][0] == 'l':
                line[0]=line[0][1:]
                lnc = line[0]
                if lnc in self.positive:
                    pos_data.append(line)
                if lnc in self.negative1:
                    neg1_data.append(line)
                if lnc in self.negative2:
                    neg2_data.append(line)
        f1.close()
        print("pos_data", len(pos_data))
        print("neg1_data", len(neg1_data))
        print("neg2_data", len(neg2_data))

        pos = pd.DataFrame(pos_data)
        pos["label"] = 1
        neg1 = pd.DataFrame(neg1_data)
        neg1["label"] = 0
        neg2 = pd.DataFrame(neg2_data)
        neg2["label"] = 0

        dataset1 = pd.concat([pos, neg1], axis=0)
        dataset2 = pd.concat([pos, neg2], axis=0)
        print("dataset1", len(dataset1))
        print("dataset2", len(dataset2))

        dataset1.to_csv('../result/human/para/1000_100_64/dataset1.csv', index=False)
        dataset2.to_csv('../result/human/para/1000_100_64/dataset2.csv', index=False)

        # dataset1.to_csv(r'E:\人类的参数文件\400_40_64\dataset1.csv', index=False)
        # dataset2.to_csv(r'E:\人类的参数文件\400_40_64\dataset2.csv', index=False)


if __name__ == '__main__':
    print("begin!")
    # human
    pos_nag = '../result/human'

    # mouse
    # pos_nag = '../result/mouse'
    obj = Pro_Data()
    obj.positive = obj.get_pos_nag(pos_nag+'/positive.csv')
    obj.negative1 = obj.get_pos_nag(pos_nag+'/negative1.csv')
    obj.negative2 = obj.get_pos_nag(pos_nag+'/negative2.csv')

    # mouse
    # dirPath = '../result/mouse/para/200_100_256'
    # obj.get_path(dirPath+'./out_mouse_200_100.txt')

    # human
    dirPath = '../result/human/para/1000_100_64'
    obj.get_path(dirPath+'/out_human_1000_100.txt')





