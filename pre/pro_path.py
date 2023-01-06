import pandas as pd
import numpy as np

class Pro_Data:
    def __init__(self):
        self.positive = []
        self.negative = []


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
        neg_data = []

        for line in f1.readlines():
            line = line.strip().split(" ")
            if line[0][0] == 'l':
                line[0]=line[0][1:]
                lnc = line[0]
                if lnc in self.positive:
                    pos_data.append(line)
                if lnc in self.negative:
                    neg_data.append(line)
        f1.close()
        print("pos_data", len(pos_data))
        print("neg_data", len(neg_data))

        pos = pd.DataFrame(pos_data)
        pos["label"] = 1
        neg = pd.DataFrame(neg_data)
        neg["label"] = 0

        dataset = pd.concat([pos, neg], axis=0)
        print("dataset1", len(dataset))

        dataset.to_csv('../result/human/para/200_100_64/dataset1.csv', index=False)

if __name__ == '__main__':
    print("begin!")
    # human
    pos_nag = '../result/human'

    # mouse
    # pos_nag = '../result/mouse'
    obj = Pro_Data()
    obj.positive = obj.get_pos_nag(pos_nag+'/positive.csv')
    obj.negative = obj.get_pos_nag(pos_nag+'/negative.csv')

    # mouse
    # dirPath = '../result/mouse/para/200_100_64'
    # obj.get_path(dirPath+'./out_mouse_200_100.txt')

    # human
    dirPath = '../result/human/para/200_100_64'
    obj.get_path(dirPath+'/out_human_200_100.txt')





