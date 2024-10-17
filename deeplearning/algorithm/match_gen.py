from itertools import combinations
from random import shuffle

teams = range(1, 9, 1)

res = list(combinations(teams, 2))
# # shuffle(res)
#
# for t in res:
#     print(t)

l1 = [(1, 2), (3, 4), (5, 6), (7, 8)]
l2 = [(5, 7), (6, 8), (1, 3), (2, 4)]
l3 = [(1, 4), (2, 3), (5, 8), (6, 7)]
l4 = [(3, 7), (4, 8), (1, 5), (2, 6)]
l5 = [(2, 5), (1, 6), (3, 8), (4, 7)]
l6 = [(3, 5), (4, 6), (1, 8), (2, 7)]
l7 = [(1, 7), (2, 8), (3, 6), (4, 5)]

day_plan = []
day_plan.append(l1)
day_plan.append(l2)
day_plan.append(l3)
day_plan.append(l4)
day_plan.append(l5)
day_plan.append(l6)
day_plan.append(l7)

# shuffle(day_plan)

for idx, dp in enumerate(day_plan):
    print("第", str(idx + 1), "轮：")
    print("第一场：", dp[:2])
    print("第二场：", dp[2:])
