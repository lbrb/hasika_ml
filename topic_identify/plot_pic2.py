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

df = pd.read_csv('result2.txt', encoding='utf-8', sep=' ')
df1 = df[(df['r'] > 0.7) & (df['p'] > 0.8)]
print(df1)
print(np.mean(df1['title_theta']))
print(np.mean(df1['content_theta']))
plt.scatter(df1['r'], df1['p'])
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
