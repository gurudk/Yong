"""

第70条 先分析性能，然后再优化

"""

from random import randint
from cProfile import Profile
from pstats import Stats
from bisect import bisect_left


def insert_sort(data):
    result = []
    for value in data:
        insert_value_bisect(result, value)
    return result


def insert_value(array, value):
    for i, existing in enumerate(array):
        if existing > value:
            array.insert(i, value)
            return
    array.append(value)


def insert_value_bisect(array, value):
    i = bisect_left(array, value)
    array.insert(i, value)


max_size = 10 ** 4
data = [randint(0, max_size) for _ in range(max_size)]
test = lambda: insert_sort(data)

profiler = Profile()
profiler.runcall(test)

stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()

print('**********************无情的分割线+++++++++++++++++++++++++++++++++++')


def my_utility(a, b):
    c = 1
    for i in range(100):
        c += a * b


def first_func():
    for _ in range(1000):
        my_utility(4, 5)


def second_func():
    for _ in range(10):
        my_utility(1, 3)


def my_program():
    for _ in range(20):
        first_func()
        second_func()


test_method_call = lambda: my_program()

profiler = Profile()
profiler.runcall(test_method_call)

stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()
stats.print_callers()
