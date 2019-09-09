import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def multiply(arr):
    result = 1
    for x in arr:
        result *= x
    return result


samples_n = 10000
bins = np.linspace(-10, 10, 20)

mu = 0
sigma = 1
samples = np.random.normal(mu, sigma, samples_n)
plt.subplot(2, 2, 1)
plt.hist(samples, bins)

p = stats.norm(mu, sigma).pdf(samples)
plt.subplot(2, 2, 3)
plt.scatter(samples, p)

x3 = np.array([1, 2, 3])
y3 = stats.norm(mu, sigma).pdf([1, 2, 3])
plt.scatter(x3, y3, color='red')
print(y3)
print(multiply(y3))

mu = 0
sigma = 3
samples = np.random.normal(mu, sigma, samples_n)
plt.subplot(2, 2, 2)
plt.hist(samples, bins)

p = stats.norm(mu, sigma).pdf(samples)
plt.subplot(2, 2, 4)
plt.scatter(samples, p)
x3 = np.array([1, 2, 3])
y3 = stats.norm(mu, sigma).pdf([1, 2, 3])
plt.scatter([1, 2, 3], stats.norm(mu, sigma).pdf([1, 2, 3]), color='red')
print(y3)
print(multiply(y3))

plt.show()
