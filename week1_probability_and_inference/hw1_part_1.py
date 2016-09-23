"""
Homework 1, part 1

Alice Hunts Dragons
"""
import numpy as np

# setup 2d-array to hold the probabilities
prob_X_Y = np.zeros((4, 3))
total = sum([x**2 + y**2 for x in [1, 2, 4] for y in [1, 3]])

for x in [1, 2, 4]:
    for y in [1, 3]:
        prob_X_Y[x - 1, y - 1] = (x**2 + y**2) / total

# setup label-mappings for easier access
row_mapping = {x: x - 1 for x in range(1, 5)}
col_mapping = {y: y - 1 for y in range(1, 4)}

# compute probability table for x
prob_X = prob_X_Y.sum(axis = 1)
prob_X_table = {x: prob_X[row_mapping[x]] for x in range(1, 5)}

# compute probability table for y
prob_Y = prob_X_Y.sum(axis = 0)
prob_Y_table = {y: prob_Y[col_mapping[y]] for y in range(1, 4)}

print(prob_X_table)
print(prob_Y_table)

