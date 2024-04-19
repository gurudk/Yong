"""

第13条 通过带*的unpacking操作来捕获多个元素，不要用切片

"""

import numpy as np

car_ages = np.random.randint(0, 20, size=10)
car_ages_descending = sorted(car_ages, reverse=True)
print(car_ages)
print(car_ages_descending)
oldest, second_oldest, *others = car_ages_descending

print(oldest, second_oldest, others)

it = iter(range(1,3))
first, second = it
print(first, ' ', second)

