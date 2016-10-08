###
# plot the Shannon-entropy for a Bernoulli random variable
###

import matplotlib.pyplot as plt
import math

def compute_bernoulli_entropy(p):
    return p * math.log(1 / p, 2) + (1 - p) * math.log(1 / (1 - p), 2)

x = np.linspace(0.01, 0.999, 100)
y = np.array([compute_bernoulli_entropy(i) for i in x])

plt.scatter(x, y)
plt.show()

