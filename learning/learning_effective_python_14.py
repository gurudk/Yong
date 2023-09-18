"""

第14条 用sort方法的key参数来表示复杂的排序逻辑

"""
import numpy as np

unsorted_numbers = np.random.randint(1, 20, size=10)

print(unsorted_numbers)

unsorted_numbers.sort()
print(unsorted_numbers)


class Tool:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return f'Tool({self.name!r}, {self.weight})'


tools = [
    Tool('level', 3.5),
    Tool('hammer', 1.25),
    Tool('screwdriver', 0.5),
    Tool('chisel', 0.25)
]

print(tools)
tools.sort(key=lambda x: x.weight, reverse=True)
print("sorted tools:", tools)


power_tools = [
    Tool('drill', 4),
    Tool('circular', 5),
    Tool('jackhammer', 40),
    Tool('sander', 4)
]

power_tools.sort(key=lambda x: (-x.weight, x.name))
print(power_tools)

power_tools.sort(key=lambda x: x.name)
power_tools.sort(key=lambda x: x.weight, reverse=True)

print(power_tools)





