# -*- coding: utf-8 -*-
"""
Working with joint probability distributions
"""
import comp_prob_inference as cp
import numpy as np

# approach 0: Do not actually represent the joint table
prob_table = {
    ('sunny', 'hot'): 3/10,
    ('sunny', 'cold'): 1/5,
    ('rainy', 'hot'): 1/30,
    ('rainy', 'cold'): 2/15,
    ('snowy', 'hot'): 0,
    ('snowy', 'cold'): 1/3
}


# approach 1: nested dicts
prob_W_T_dict = {}
for w in {'sunny', 'rainy', 'snowy'}:
    prob_W_T_dict[w] = {}

prob_W_T_dict['sunny']['hot'] = 3/10
prob_W_T_dict['sunny']['cold'] = 1/5
prob_W_T_dict['rainy']['hot'] = 1/30
prob_W_T_dict['rainy']['cold'] = 2/15
prob_W_T_dict['snowy']['hot'] = 0
prob_W_T_dict['snowy']['cold'] = 1/3

cp.print_joint_prob_table_dict(prob_W_T_dict)


# approach 2: 2d-arrays
prob_W_T_rows = ['sunny', 'rainy', 'snowy']
prob_W_T_cols = ['hot', 'cold']

# create mappings for faster index-access
prob_W_T_row_mapping = {
    label: index for index, label in enumerate(prob_W_T_rows)
}
prob_W_T_col_mapping = {
    label: index for index, label in enumerate(prob_W_T_cols)
}
prob_W_T_array = np.array([[3/10, 1/5], [1/30, 2/15], [0, 1/3]])

cp.print_joint_prob_table_array(prob_W_T_array, prob_W_T_rows, prob_W_T_cols)




