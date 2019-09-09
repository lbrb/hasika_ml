import matplotlib.pyplot as plt
import numpy as np

samples_n = 10000
bins = np.linspace(-10, 10, 20)

mu = 0
sigma = 1
samples = np.random.normal(mu, sigma, samples_n)
plt.subplot(1, 2, 1)
plt.hist(samples, bins)

mu = 0
sigma = 3
samples = np.random.normal(mu, sigma, samples_n)
plt.subplot(1, 2, 2)
plt.hist(samples, bins)
plt.show()
