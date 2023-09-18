"""

第6条 将数据结构拆分到多个变量里，不要专门通过下标访问

"""

import numpy as np

snack_calories = {
    'chips': 140,
    'popcorn': 80,
    'nuts':190
}
items = tuple(snack_calories.items())
print(items)

item = ('Peanut butter', 'Jelly')
# first = item[0]
# second = item[1]
first, second = item #unpacking

print(first, 'and', second)


def bubble_sort(a):
    for _ in range(len(a)):
        for i in range(1, len(a)):
            if a[i] < a[i-1]:
                a[i-1], a[i] = a[i], a[i-1] #swqp, nice，居然可以如此使用。


test_arr = np.random.randint(10, size=(10))
print(test_arr)
bubble_sort(test_arr)
print(test_arr)

snacks = [('bacon', 350), ('donut', 240), ('muffin', 190)]
for rank, (name, calories) in enumerate(snacks, 1): #enumerate 的序号从什么序数开始，第二个变量
    print(f'#{rank}: {name} has {calories} calories')








