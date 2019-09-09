import numpy as np
import matplotlib.pyplot as plt

# f(x) = aX + b
X = np.linspace(-100, 100, 50)
y = 2 * X + 1 + np.random.normal(0, 20, 50)
y_hat = 2 *X +1
plt.scatter(X, y)
plt.plot(X, y_hat)
plt.show()
