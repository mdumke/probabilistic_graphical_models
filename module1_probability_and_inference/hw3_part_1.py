"""
Ainsley works on problem sets
"""
from math import *

# implements the binomial coefficient
def choose(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

# returns the probability of learning a given number of concepts
def p_concepts_given_psets(c, s):
    assert c <= 2 * s, "e1: Cannot learn more than 2s concepts"
    return 1 / (2 * s + 1)

# returns the probability of having d drinks when solving s problems
# q is the prob of having a drink after solving a problem
def p_drinks_given_psets(d, s, q):
    assert d <= s, "e3: Cannot have more drinks than problems"
    return choose(s, d) * q**d * (1 - q)**(s - d)

# returns the marginal prbability that A had d drinks
def p_drinks(d, q):
    total = 0
    for s in range(d, 5):
        total += p_psets(s) * p_drinks_given_psets(d, s, q)
    return total

# returns the marginal probability that A solved s psets
def p_psets(s):
    return 1 / 4

# returns the probability that A solved s problems if we find d drinks
def p_psets_given_drinks(s, d, q):
    return p_psets(s) * p_drinks_given_psets(d, s, q) / p_drinks(d, q)

# returns the expactation for how many concepts A learned if she solved s psets
def e_concepts_given_psets(s):
    total = 0
    for c in range(0, 2 * s + 1):
        total += c * p_concepts_given_psets(c, s)
    return total

# returns the expectation for how many concepts A learned if we find d drinks
def e_concepts_given_drinks(d, q):
    total = 0
    for s in range(d, 5):
        total += e_concepts_given_psets(s) * p_psets_given_drinks(s, d, q)
    return total

print(e_concepts_given_drinks(1, 0.2))
print(e_concepts_given_drinks(2, 0.5))
print(e_concepts_given_drinks(3, 0.7))

# some sanity checks
assert e_concepts_given_psets(4) > e_concepts_given_psets(1), '001'
assert e_concepts_given_drinks(3, 0.2) > e_concepts_given_drinks(2, 0.2), '002'
assert e_concepts_given_drinks(2, 0.2) > e_concepts_given_drinks(2, 0.8), '003'

# after getting some correct results:
assert e_concepts_given_drinks(1, 0.2) - 2.7637028 < 1e-6, '004'
assert e_concepts_given_drinks(2, 0.5) - 3.125 < 1e-6, '005'
assert e_concepts_given_drinks(3, 0.7) - 3.5454545 < 1e-6, '006'

