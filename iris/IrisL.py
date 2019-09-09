from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris.data
target = iris.target
print("features.shape: ", features.shape)
print("features: ", features)
print("target.shape：", target.shape)
print("target:", target)

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()

plt.scatter(features[:, 0], features[:, 2], c=target)
plt.show()

plt.scatter(features[:, 0], features[:, 3], c=target)
plt.show()

plt.scatter(features[:, 1], features[:, 2], c=target)
plt.show()

plt.scatter(features[:, 1], features[:, 3], c=target)
plt.show()

plt.scatter(features[:, 2], features[:, 3], c=target)
plt.show()

