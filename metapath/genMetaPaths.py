#!/home/yyzhang/anaconda3/bin/python3.6
#这是生成路径的文件
import os
import sys
import random


class MetaPathGenetor:
    def __init__(self):
        self.id_lncRNA = dict() # lncRNA的id
        self.id_protein = dict() # protein 的id
        # 存储了lncRNA-protein 关系的2组数据
        self.lncRNA_proteinlist = dict()
        self.protein_lncRNAlist = dict()
        # 生成路径要用的2组 数据
        self.lnc_pro = dict()
        self.pro_lnc = dict()


    def display_data(self):
        print(len(self.lncRNA_proteinlist))
        print(len(self.protein_lncRNAlist))

    def read_data(self,dirPath):
        with open(dirPath+'/id_lncRNA.txt',encoding='utf-8') as lncfile:

            for line in lncfile.readlines():
                data = line.strip().split("\t")
                if len(data) == 2:
                    num,lnc = data[0],data[1]
                    self.id_lncRNA[num] = lnc
        
        with open(dirPath+'/id_protein.txt',encoding='gbk') as profile:
            for line in profile.readlines():
                data = line.strip().split("\t")
                if len(data) == 2:
                    num,pro = data[0],data[1]
                    self.id_protein[num] = pro

        
        with open(dirPath+'/id_lncRNA_protein.txt',encoding='utf-8') as lpfile:
            for line in lpfile.readlines():
                data = line.strip().split("\t")
                if len(data) == 2:
                    l,p = data[0],data[1]

                    if l not in self.lncRNA_proteinlist:
                        self.lncRNA_proteinlist[l] = []
                    self.lncRNA_proteinlist[l].append(p)

                    if p not in self.protein_lncRNAlist:
                        self.protein_lncRNAlist[p] = []
                    self.protein_lncRNAlist[p].append(l)
        
    def generate_random_lpl(self,dirPath,outfileName,numWalks,walkLength):
        outfile = open(dirPath+'/'+outfileName,'w',encoding = "utf-8")
        for lnc in self.lncRNA_proteinlist:
            lnc0 = lnc
            for j in range(0,numWalks):
                outline = self.id_lncRNA[lnc0]
                for i in range(0,walkLength):
                    pros = self.lncRNA_proteinlist[lnc]
                    nump = len(pros)
                    proId = random.randrange(nump)
                    pro = pros[proId]

                    outline +="\t"+self.id_protein[pro]
                    lncs = self.protein_lncRNAlist[pro]
                    numl = len(lncs)
                    lncId = random.randrange(numl)
                    lncrna = lncs[lncId]
                    #print(lncrna)
                    outline +="\t"+self.id_lncRNA[lncrna]
                outfile.write(outline +'\n')
        outfile.close()
#./genMetaPaths.py 1000 100 ../data/mouse  in_mouse_1000_10.txt
#一些从键盘输入的参数
#每个节点生成几个句子
numWalks = int(sys.argv[1])

#路径长度
walkLength = int(sys.argv[2])

# 数据文件的路径
dirPath = sys.argv[3]
#导出的文件名字
outfileName = sys.argv[4]
if __name__ == "__main__":
    print("begin!")
    obj = MetaPathGenetor()
    obj.read_data(dirPath)
    obj.display_data()
    obj.generate_random_lpl(dirPath,outfileName, numWalks, walkLength)
    print("end!")
