import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel(r"C:\Users\zyy\Desktop\code\prediction esslnc\result\mouse\SVM_para.xlsx",usecols=["dataset","AUROC","AUPR"])
M2 = data[data["dataset"]=="M2"]
auroc = list(M2["AUROC"])
aupr = list(M2["AUPR"])
x_data = np.arange(11)
x = ["1_0.1_rbf","1_0.01_rbf","10_0.1_rbf","10_0.01_rbf","10_0.001_rbf","100_0.1_rbf","100_0.01_rbf","100_0.001_rbf", "1_linear", "10_linear","100_linear"]

bar_width = 0.35
plt.bar(x_data, aupr, bar_width, align="center", color="#E2C3C9", label='AUPR')
plt.bar(x_data+bar_width, auroc, bar_width, align="center",color="#80AFBF", label='AUROC')

# for a,b in zip(x_data,aupr):   #柱子上的数字显示
#     plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=5);
# for a,b in zip(x_data+bar_width,auroc):
#     plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=5);


plt.xticks(rotation=30)

plt.ylim((0, 1))
plt.legend(loc="lower right")
plt.ylabel('scores(LOOCV)')
plt.title("M2 dataset")

plt.xticks(x_data+bar_width/2, x)
# plt.show()
# 设置tight bbox
plt.savefig('../img/SVM_M2.png', bbox_inches='tight')
