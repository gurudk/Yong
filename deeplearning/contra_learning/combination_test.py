from itertools import combinations

idxs = range(9)

res = list(combinations(idxs, 2))
for r in res:
    print(r)
