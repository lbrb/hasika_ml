import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pandas as pd

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

keys = []
value1s = []
value2s = []
f1s = []
result_dict = {}

df = pd.read_csv('result.csv')
group_df = df.groupby(by=['multi_title', 'n_keywords'])

def check(multi_title, n_keywords):
    return n_keywords in [16, 21]

for (k1, k2), group in group_df:
    print(k1, k2)
    print(group)
    if check(k1, k2):
        plt.scatter(list(group['r']), list(group['p']), label=str(k1) + str(k2))
    print('-' * 40)
plt.xlabel('查全率')
plt.ylabel('查准率')
plt.title('P-R曲线')
plt.xlim(-0.01, 1.02)
plt.ylim(-0.01, 1.02)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='lower left', fancybox=True, framealpha=0.8, fontsize=12)
plt.show()

for (k1, k2), group in group_df:
    print(k1, k2)
    print(group)
    if check(k1, k2):
        plt.scatter(list(group['theta']), list(group['p']), label=str(k1) + str(k2))
    print('-' * 40)
plt.xlabel('相似度阈值')
plt.ylabel('查准率')
plt.title('P-相似度阈值 曲线')
plt.xlim(-0.01, 1.02)
plt.ylim(-0.01, 1.02)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='lower left', fancybox=True, framealpha=0.8, fontsize=12)
plt.show()


for (k1, k2), group in group_df:
    print(k1, k2)
    print(group)
    if check(k1, k2):
        plt.scatter(list(group['theta']), list(group['r']), label=str(k1) + str(k2))
    print('-' * 40)
plt.xlabel('相似度阈值')
plt.ylabel('查全率')
plt.title('R-相似度阈值 曲线')
plt.xlim(-0.01, 1.02)
plt.ylim(-0.01, 1.02)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(loc='lower left', fancybox=True, framealpha=0.8, fontsize=12)
plt.show()
