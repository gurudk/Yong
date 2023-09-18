"""

第54条 利用Lock防止多个线程争用同一份数据

"""


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self, offset):
        self.count += offset


def worker(sensor_index, how_many, counter):
    for _ in range(how_many):
        counter.increment(1)

