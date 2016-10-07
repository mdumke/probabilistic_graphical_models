import numpy as np

l = np.zeros((4, 4))

for y in range(4):
    for x in range(4):
        if y == x:
            l[y, x] = 2
        else:
            l[y, x] = 1 / abs(y - x)

alphas = 1 / l.sum(axis = 0)

l2 = np.zeros((4, 4))

for y in range(4):
    for x in range(4):
        if y == x:
            l2[y, x] = 2 * alphas[x]
        else:
            l2[y, x] = alphas[x] / abs(y - x)
