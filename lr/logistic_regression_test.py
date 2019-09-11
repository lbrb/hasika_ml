import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lr.hasika_logistic_regression import HasikaLogisticRegression
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 样本
    X = np.array([[1, 2], [2, 4], [5, 3.02], [10, 5.01]])
    # X = np.array([1, 2, 5, 10]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    print(X.shape)
    print(y.shape)

    # 轮子
    liner = HasikaLogisticRegression()
    liner.fit(X, y)

    # scikit-learn 类库
    liner2 = LogisticRegression()
    liner2.fit(X, y)

    # 画轮子
    x1 = np.linspace(0, 10, 1000)
    x2 = np.linspace(0, 10, 1000)
    x1, x2 = np.meshgrid(x1, x2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    y_pret = liner.predict(x_show)
    print('轮子 回归系数, 截距')
    print(liner.coef_, liner.intercept_)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    print("轮子：", y_pret.shape)
    plt.pcolormesh(x1, x2, y_pret.reshape(x1.shape), cmap=cm_light)
    print("轮子：", x1.shape)
    print("轮子：", y_pret.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark)

    plt.show()

    # 画scikit-learn类库函数线
    x1 = np.linspace(0, 10, 1000)
    x2 = np.linspace(0, 10, 1000)
    x1, x2 = np.meshgrid(x1, x2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    y_pret = liner2.predict(x_show)
    print('scikit-learn类库 回归系数, 截距')
    print(liner2.coef_, liner2.intercept_)
    # b2 = liner2.intercept_ + a * liner2.coef_

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    print("类库：", y_pret.shape)
    plt.pcolormesh(x1, x2, y_pret.reshape(x1.shape), cmap=cm_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark)
    print("类库：", x1.shape)
    print("类库：", y_pret.shape)
    plt.show()
