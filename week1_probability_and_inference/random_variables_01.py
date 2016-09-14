# -*- coding: utf-8 -*-
"""
Random variables as mappings
"""
import comp_prob_inference

prob_space = {'sunny': 1/2, 'rainy': 1/6, 'snowy': 1/3}

def run_experiment():
    random_outcome = comp_prob_inference.sample_from_finite_probability_space(prob_space)
    W = random_outcome
    if random_outcome == 'sunny':
        I = 1
    else:
        I = 0
    print(W, I)

for i in range(1, 100):
    run_experiment()
