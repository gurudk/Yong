"""

第36条 考虑用itertools拼装迭代器和生成器

"""
import itertools

values = list(range(1,11,1))
less_then_seven = lambda x: x < 7
it = itertools.dropwhile(less_then_seven, values)
print(list(it))

it = itertools.takewhile(less_then_seven, values)
print(list(it))

