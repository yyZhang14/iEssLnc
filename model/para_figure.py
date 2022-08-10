import matplotlib.pyplot as plt

# Mouse
# node【x, 100 , 64】
# dataset1_AUROC = [0.931, 0.952, 0.942, 0.936, 0.939]
# dataset1_AUPR = [0.959, 0.97, 0.966, 0.96, 0.963]
# dataset2_AUROC = [0.914, 0.917, 0.913, 0.906, 0.926]
# dataset2_AUPR = [0.887, 0.888, 0.886, 0.888, 0.89]
# x = ['200', '300', '400', '500', '1000']

# length [300, x, 64]
# dataset1_AUROC = [0.946, 0.952, 0.950,  0.952, 0.931]
# dataset1_AUPR = [0.964, 0.963, 0.965,  0.97, 0.962]
# dataset2_AUROC = [0.919, 0.926, 0.93, 0.917, 0.908]
# dataset2_AUPR = [0.904, 0.907, 0.902,  0.888, 0.891 ]
# x = ['30', '40', '50', '100', '200']

# vector [300 50 x]
# dataset1_AUROC = [0.957, 0.973, 0.95, 0.95, 0.95]
# dataset1_AUPR = [0.971, 0.979, 0.965, 0.967, 0.968]
# dataset2_AUROC = [0.93, 0.952, 0.93, 0.938, 0.936]
# dataset2_AUPR = [0.907, 0.915, 0.902, 0.907, 0.899]
# x = ['16', '32', '64', '128', '256']

# Human
# node【x, 100, 64】
dataset1_AUROC = [0.879, 0.905, 0.903, 0.907, 0.899]
dataset1_AUPR = [0.848, 0.915, 0.915, 0.92, 0.904]
dataset2_AUROC = [0.878, 0.908, 0.9, 0.901, 0.903]
dataset2_AUPR = [0.773, 0.863, 0.861, 0.859, 0.86]
x = ['200', '300', '400', '500', '1000']

# length [300, x, 64]
# dataset1_AUROC = [0.913, 0.895, 0.892, 0.905, 0.911]
# dataset1_AUPR = [0.923, 0.901, 0.898, 0.915, 0.916]
# dataset2_AUROC = [0.908, 0.899, 0.897, 0.908, 0.908]
# dataset2_AUPR = [0.87, 0.863, 0.858, 0.863, 0.863]
# x = ['30', '40', '50', '100', '200']

# vector [300 30 x]
# dataset1_AUROC = [0.903, 0.906, 0.913, 0.914, 0.912]
# dataset1_AUPR = [0.914, 0.916, 0.923, 0.924, 0.921]
# dataset2_AUROC = [0.902, 0.906, 0.908, 0.911, 0.907]
# dataset2_AUPR = [0.851, 0.865, 0.87, 0.87, 0.873]
# x = ['16', '32', '64', '128', '256']

plt.plot(x, dataset1_AUROC, color="#A0D995", marker='o', markerfacecolor='white',  linewidth=1.5, label='H1_AUROC')
plt.plot(x, dataset1_AUPR, color="#3AB0FF", marker='o', markerfacecolor='white',  linewidth=1.5, label='H1_AUPR')
plt.plot(x, dataset2_AUROC, color="#F87474", marker='o', markerfacecolor='white',  linewidth=1.5, label='H2_AUROC')
plt.plot(x, dataset2_AUPR, color="#FFEA11", marker='o', markerfacecolor='white', linewidth=1.5, label='H2_AUPR')

plt.ylim((0.5, 1))
plt.legend(loc="lower right")
plt.xlabel('The number of walks of per node n')
# plt.xlabel('The length of each walk l')
# plt.xlabel('The dimension of embedding representation r')
plt.ylabel('scores(LOOCV)')
plt.title("Human")

# plt.title("Mouse")
# plt.show()
plt.savefig('../img/node1.png', bbox_inches='tight')
# plt.savefig('../img/length1.png', bbox_inches='tight')
# plt.savefig('../img/vector1.png', bbox_inches='tight')
