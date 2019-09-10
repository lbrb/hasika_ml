import numpy as np


class HasikaLinearRegression:
    def __init__(self):
        # theta 函数的参数
        self.theta = np.zeros(1)
        # 样本
        self.X = np.zeros(1)
        # 标记
        self.y = np.zeros(1)
        # 学习率
        self.alpha = 0.01
        # 回归系数
        self.coef_ = np.zeros(1)
        # 截距
        self.intercept_ = np.zeros(1)

    def fit(self, X, y):
        x_0 = np.ones(X.shape[0])
        self.X = np.insert(X, 0, values=x_0, axis=1)
        self.theta = np.ones(self.X.shape[1]).reshape(1, -1)
        self.y = y
        print("X: ", X)
        print("y: ", y)
        print("theta: ", self.theta)
        self.loop()
        self._set_intercept()

    # 不断的迭代学习
    def loop(self):
        loop_n = 0
        while loop_n < 1000:
            gradient = self.get_gradient()
            # print("gradient: ", gradient)
            self.theta -= self.alpha * gradient
            print("theta: ", self.theta)
            loop_n += 1

    # 获取梯度
    def get_gradient(self):
        h_theta = np.dot(self.X, self.theta.reshape(-1, 1))
        # print("h_theta: ", h_theta)
        loss = h_theta - self.y.reshape(-1, 1)
        # print("loss: ", loss)
        gradient = np.dot(loss.T, self.X)
        return gradient

    # 设置回归系数及截距
    def _set_intercept(self):
        self.coef_ = self.theta[0][1:]
        self.intercept_ = self.theta[0][0]
