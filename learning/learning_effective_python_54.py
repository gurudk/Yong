"""

第54条 利用Lock防止多个线程争用同一份数据

"""

from threading import Thread
from threading import Lock


class Counter:
    def __init__(self):
        self.count = 0
        self.lock = Lock()

    def increment(self, offset):
        with self.lock:
            self.count += offset


def worker(sensor_index, how_many, counter):
    for _ in range(how_many):
        counter.increment(1)


how_many = 10**6
counter = Counter()

threads = []

for i in range(5):
    thread = Thread(target=worker, args=(i, how_many, counter))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

expected = how_many * 5

found = counter.count
print(f'Counter should be {expected}, got {found}')


