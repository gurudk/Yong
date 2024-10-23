import random
import itertools

random.seed(0)

ll = list(range(10))

print(random.sample(ll, 10))
# [6, 9, 0, 2, 4, 3, 5, 1, 8, 7]

print(random.choices(ll, k=10))
# [5, 9, 5, 2, 7, 6, 2, 9, 9, 8]

print(list(itertools.combinations(ll, 2)))

print(round(254 * (0.6, 0.2, 0.2)[1]))
