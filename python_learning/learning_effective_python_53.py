"""

第53条 可以用线程执行阻塞式IO，但不要用它做并行计算。

"""

import time
import select
import socket

from threading import Thread


def factorize(number):
    for i in range(1, number + 1):
        if number % i == 0:
            yield i


def is_factual(number):
    result = True
    for i in range(2, int(number/2)):
        if number % i == 0:
            result = False
            break

    return result


def factor_generator(start, end):
    for i in range(start, end):
        if is_factual(i):
            yield i


print(list(factor_generator(20000, 20100)))


numbers = [2139079, 1214759, 1516637, 1852285]

start = time.time()
for number in numbers:
    list(factorize(number))

end = time.time()

delta = end - start
print(f'[SingleThread]Took {delta:3f} seconds')


class FactorizeThread(Thread):
    def __init__(self, number):
        super().__init__()
        self.number = number

    def run(self):
        self.factors = list(factorize(number))


print('Multi thread to get factors for a number =============================================begin')

start = time.time()

threads = []
for number in numbers:
    thread = FactorizeThread(number)
    thread.start()
    threads.append(thread)


for thread in threads:
    thread.join()

end = time.time()
delta = end - start
print(f'[MultiThread]Tool {delta:3f} seconds')

print('Multi thread to get factors for a number =============================================end')

print(f'[MultiThread]socket io 阻塞示例 begin..................j..................................')


def slow_systemcall():
    select.select([socket.socket()], [], [], 0.1)


start = time.time()
for _ in range(5):
    slow_systemcall()

end = time.time()
delta = end - start
print(f'[singleThread] socket io took {delta: 3f} seconds')

start = time.time()

threads = []
for _ in range(5):
    thread = Thread(target=slow_systemcall())
    thread.start()
    threads.append(thread)

def compute_helicopter_location(index):
    ...


for i in range(5):
    compute_helicopter_location(i)


for thread in threads:
    thread.join()

end = time.time()

delta = end - start

print(f'[MulitiThread] Took {delta} seconds')

print(f'[MultiThread]socket io 阻塞示例 end..................j..................................')


