###
# exercise: probability of using a certain mode of transportation
###

import numpy as np

# given independent r.v.s R (rain) and C (cold)
p_R = np.array([0.5, 0.5])
p_C = np.array([0.5, 0.5])

# from the problem description extract the joint probability distribution
p_C_R_T_W = np.array([
    [[[0, 0.25],
      [0, 0]],
     [[0, 0],
      [0.25, 0]]],
    [[[0, 0],
      [0.25, 0]],
     [[0, 0.25],
      [0, 0]]]])

# from the joint probability we can compute a range of others
p_R_T_W = np.sum(p_C_R_T_W, axis = 0)
p_T_W_given_R = p_R_T_W / p_R

p_R_T = np.sum(p_R_T_W, axis = 2)
p_R_W = np.sum(p_R_T_W, axis = 1)
p_T = np.sum(p_R_T, axis = 0)

p_T_given_R = p_R_T / p_R
p_W_given_R = p_R_W / p_R

p_R_T_W_given_C = p_C_R_T_W / p_C
p_T_W_given_C_R = p_R_T_W_given_C / p_R

p_T_given_C_R = np.sum(p_T_W_given_C_R, axis = 3)
p_W_given_C_R = np.sum(p_T_W_given_C_R, axis = 2)

p_C_R_W = np.sum(p_C_R_T_W, axis = 2)
p_C_W = np.sum(p_C_R_W, axis = 1)
p_W = np.sum(p_R_W, axis = 0)

p_C_R_given_W = p_C_R_W / p_W
p_R_given_W = p_R_W / p_W
p_C_given_W = p_C_W / p_W

# checking independencies
print("W ind. T | R?")
for r in [0, 1]:
    for t in [0, 1]:
        for w in [0, 1]:
            print(p_T_given_R[r][t] * p_W_given_R[r][w], p_T_W_given_R[r][t][w])

print("W ind. T | R, C?")
for c in [0, 1]:
    for r in [0, 1]:
        for t in [0, 1]:
            for w in [0, 1]:
                print(p_T_given_C_R[c][r][t] * p_W_given_C_R[c][r][w], p_T_W_given_C_R[c][r][t][w])

print("R ind. C | W?")
for c in [0, 1]:
    for r in [0, 1]:
        for w in [0, 1]:
            print(p_R_given_W[r][w] * p_C_given_W[c][w], p_C_R_given_W[c][r][w])

print("R ind. T?")
for r in [0, 1]:
    for t in [0, 1]:
        print(p_R[r] * p_T[t], p_R_T[r][t])

# searching for missing indepencencies
p_C_R = np.sum(np.sum(p_C_R_T_W, axis = 3), axis = 2)
p_C_T = np.sum(np.sum(p_C_R_T_W, axis = 3), axis = 1)
p_T_W = np.sum(np.sum(p_C_R_T_W, axis = 1), axis = 0)

