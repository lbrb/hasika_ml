import numpy as np
import pandas as pd

path = 'output_09_24_simple.xls'
pd1 = pd.read_excel(path).dropna()
pd2 = pd1[pd1['正文内容'].str.contains('"type":"text"')]
print(len(pd2))
print(pd2)

