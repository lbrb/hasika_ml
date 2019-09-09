import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
plt.scatter(1, stats.norm(mu, sigma).pdf(1), color='red')

mu = 0
sigma = 3
samples = np.random.normal(mu, sigma, samples_n)
plt.subplot(2, 2, 2)
plt.hist(samples, bins)

p = stats.norm(mu, sigma).pdf(samples)
plt.subplot(2, 2, 4)
plt.scatter(samples, p)
plt.scatter(1, stats.norm(mu, sigma).pdf(1), color='red')

plt.show()
