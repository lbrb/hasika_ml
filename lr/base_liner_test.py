import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lr.base_liner import HasikaLinearRegression
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 样本
    X = np.array([0.49, 1.01, 3.02, 5.01])
    X = X.reshape(-1, 1)
    y = np.array([0.51, 1.50, 4.01, 4.81])

    # 轮子
    liner = HasikaLinearRegression()
    liner.fit(X, y)

    # scikit-learn 类库
    liner2 = LinearRegression()
    liner2.fit(X, y)

    # 画样本点
    plt.scatter(X, y, c='r', label='样本')

    # 画轮子函数线
    a = np.linspace(0, 10, 100)
    print('轮子 回归系数, 截距')
    print(liner.coef_, liner.intercept_)
    b1 = liner.intercept_ + a * liner.coef_
    plt.plot(a, b1, c='g', label='轮子')

    # 画scikit-learn类库函数线
    print('scikit-learn类库 回归系数, 截距')
    print(liner2.coef_, liner2.intercept_)
    b2 = liner2.intercept_ + a * liner2.coef_
    plt.plot(a, b2, c='b', label='scikit-learn类库')

    plt.legend(loc='upper left')
    plt.show()
