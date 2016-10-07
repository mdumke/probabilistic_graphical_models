# -*- coding: utf-8 -*-
"""
Probability mass function of the sum of two fair dice
"""

# build basic probability space
prob_space = {}

for i in range(1, 7):
    for j in range(1, 7):
        prob_space[(i, j)] = 1/36

# initialize pmf
pmf = {}

for i in range(2, 13):
    pmf[i] = 0

# compute mapping
for outcome in prob_space:
    pmf[outcome[0] + outcome[1]] += prob_space[outcome]

print(pmf)

