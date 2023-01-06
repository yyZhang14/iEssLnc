import pandas as pd

def read_predata(dirPath):
    data =pd.read_csv(dirPath+'/LPI.csv',encoding="gbk")
    lnc=set(data["lncRNA"])
    pro=set(data["protein"])
    l=list(data["lncRNA"])
    p=list(data["protein"])

    print(len(data),len(lnc),len(pro))
    id_lnc={}
    id_pro={}

    lnc_id={}
    pro_id={}
    j=0
    n= len(lnc)+1
    for i in lnc:
        id_lnc[j]=i
        lnc_id[i]=j
        j=j+1
    for k in pro:
        id_pro[n]=k
        pro_id[k]=n
        n=n+1
    
    f1=open(dirPath+'/id_lncRNA.txt',"w")
    for key,value in id_lnc.items():
        data2=str(key)+"\t"+"l"+str(value)+"\n"
        f1.write(data2)
    f1.close()

    f2=open(dirPath+'/id_protein.txt',"w")
    for key,value in id_pro.items():
        data3=str(key)+"\t"+"p"+str(value)+"\n"
        f2.write(data3)
    f2.close()

    with open(dirPath+'/id_lncRNA_protein.txt',"w") as lp_file:
        for i in range(len(data)):
            lnc1=str(l[i])
            pro1=str(p[i])
            lp=str(lnc_id[lnc1])+"\t"+str(pro_id[pro1])+"\n"
            lp_file.write(lp)
    lp_file.close()  


dirPath = '../data/human'
if __name__ == "__main__":
    print("begin!")
    read_predata(dirPath)
    print("end!")
