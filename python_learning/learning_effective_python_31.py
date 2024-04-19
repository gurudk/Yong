"""
第31条：谨慎地迭代函数所收到的参数
"""
from collections.abc import Iterator


def normalize(numbers):
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value/total
        result.append(percent)
    return result


visits = [15, 35, 80]
percentages = normalize(visits)
print(percentages)


def read_visits(data_path):
    with open(data_path) as f:
        for line in f:
            yield int(line)


it = read_visits("./my_numbers.txt")
print(list(it))
print(list(it))
# percentages = normalize(it)
# print(percentages)


def normalize_copy(numbers):
    numbers_copy = list(numbers)
    total = sum(numbers_copy)
    result = []
    for value in numbers_copy:
        percent = 100 * value / total
        result.append(percent)
    return result


it_copy = read_visits("./my_numbers.txt")
percentages = normalize_copy(it_copy)
print(percentages)


def normalize_func(get_iter):
    total = sum(get_iter())
    result = []
    for value in get_iter():
        percent = 100 * value / total
        result.append(percent)
    return result


percentages = normalize_func(lambda: read_visits("./my_numbers.txt"))
print(percentages)


class ReadVisits:
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                yield int(line)


visits_container = ReadVisits("./my_numbers.txt")
percentages = normalize(visits_container)
print(percentages)


def normalize_defensive(numbers):
    if iter(numbers) is numbers:
        raise TypeError('Must supply a container')
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result


visits = [15, 35, 80]
percentages = normalize_defensive(visits)
print(percentages)


def normalize_defensive_abc(numbers):
    if isinstance(numbers, Iterator):
        raise TypeError('Must supply a container')
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result


# 没有实现__iter__方法，不是迭代容器，所以会报错
visits = [15, 35, 80]
it_n = iter(visits)
percentages = normalize_defensive_abc(it_n)
print(percentages)