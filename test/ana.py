import numpy as np
import pandas as pd

df = pd.read_excel('分布聚合.xlsx')
clickgroup = df.groupby('点击').agg({'人数':np.sum})
print(clickgroup)

precents = [0.90, 0.95, 0.98, 0.99]

precent_sum = clickgroup['人数'].quantile(precents)
print(precent_sum)

# for index, count in clickgroup.iterrows():

