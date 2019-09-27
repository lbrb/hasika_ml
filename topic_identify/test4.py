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
# with open('result.csv', mode='r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         key = line.split(': ')[0]
#         value_tupple = line.split(': ')[1].split(',')
#         value1 = float(value_tupple[0][1:].strip())
#         value2 = float(value_tupple[1][:-2].strip())
#         keys.append(key)
#         value1s.append(value1)
#         value2s.append(value2)
#         f1 = 0.5*(1/value1 + 1/value2)
#         f1s.append(f1)
#         result_dict[key+str(value1)+str(value2)] = f1

df = pd.read_csv('result.csv')
group_df = df.groupby(by=['multi_title', 'n_keywords'])
for (k1, k2), group in group_df:
    print(k1, k2)
    print(group)
    if k2 in [21]:
        plt.plot(list(group['p']), list(group['r']), label=str(k1) + str(k2))
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
#
# result_dict1 = sorted(result_dict.items(), key=lambda item: item[1], reverse=True)
