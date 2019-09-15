import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from dt.hasika_decisin_tree_classifier import HasikaDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def show_pic(X, y, dt):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def get_dataset():
    # 数据集
    # 特征1
    dataSet = np.array([['青年', '否', '否', '一般', '不同意'],
                        ['青年', '否', '否', '好', '不同意'],
                        ['青年', '是', '否', '好', '同意'],
                        ['青年', '是', '是', '一般', '同意'],
                        ['青年', '否', '否', '一般', '不同意'],
                        ['中年', '否', '否', '一般', '不同意'],
                        ['中年', '否', '否', '好', '不同意'],
                        ['中年', '是', '是', '好', '同意'],
                        ['中年', '否', '是', '非常好', '同意'],
                        ['中年', '否', '是', '非常好', '同意'],
                        ['老年', '否', '是', '非常好', '同意'],
                        ['老年', '否', '是', '好', '同意'],
                        ['老年', '是', '否', '好', '同意'],
                        ['老年', '是', '否', '非常好', '同意'],
                        ['老年', '否', '否', '一般', '不同意']])
    # 特征集
    labels = ['年龄', '有工作', '有房子', '信贷情况', '是否同意贷款']
    return dataSet, labels


if __name__ == '__main__':
    # 样本
    data_set, labels = get_dataset()
    encoder = LabelEncoder()
    # data = pd.DataFrame(data_set)
    for i in range(data_set.shape[1]):
        encoder.fit(data_set[:, i])
        data_set[:, i] = encoder.transform(data_set[:, i])
    X = data_set[:, :-1]
    y = data_set[:, -1]
    print(X)
    print(y)


    dt1 = HasikaDecisionTreeClassifier()
    dt1.fit(X, y)
    print(dt1.print_node())

    print(dt1.predict(X))
