import numpy as np


class HasikaLogisticRegression:
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

    def predict(self, X):
        x_0 = np.ones(X.shape[0])
        X = np.insert(X, 0, values=x_0, axis=1)
        pred = self.get_h_theta(X).flatten()
        # print("pred: ", pred)
        pred = np.array(list(map(self.classify, pred)))
        # print("pred: ", pred)
        return pred

    def loop(self):
        i = 0
        while i < 10000:
            gradient = self.get_gradient()
            # print("gradient: ", gradient)
            self.theta -= self.alpha * gradient
            print("theta: ", self.theta)
            i += 1

    def get_gradient(self):
        h_theta = self.get_h_theta(self.X)
        loss = h_theta - self.y.reshape(-1, 1)
        # print("loss: ", loss)
        gradient = np.dot(loss.T, self.X)
        return gradient

    def get_h_theta(self, X):
        linear_theta = np.dot(X, self.theta.reshape(-1, 1))
        # print("linear_theta: ", linear_theta)
        h_theta = self.sigmoid(linear_theta)
        # print("h_theta: ", h_theta)
        return h_theta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _set_intercept(self):
        self.intercept_ = 1
        self.coef_ = 1

    def classify(self, x):
        if x > 0.5:
            return 1
        else:
            return 0
