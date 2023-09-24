

from queue import Queue
from threading import Thread


class ClosableQueue(Queue):
    SENTINEL = object()

    def close(self):
        self.put(self.SENTINEL)

    def __iter__(self):
        while True:
            item = self.get()
            try:
                if item is self.SENTINEL:
                    return
                yield item
            finally:
                self.task_done()


class StoppableWorker(Thread):
    def __init__(self, func, in_queue, out_queue):
        super().__init__()
        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        for element in self.in_queue:
            result = self.func(element)
            self.out_queue.put(result)


def start_threads(count, *args):
    threads = [StoppableWorker(*args) for _ in range(count)]
    for thread in threads:
        thread.start()
    return threads


def stop_threads(closable_queue, threads):
    for _ in threads:
        closable_queue.close()

    closable_queue.join()

    for thread in threads:
        thread.join()


def calculate(num):
    return num * num


inq = ClosableQueue()
outq = ClosableQueue()

calc_threads = start_threads(3, calculate, inq, outq)

batch = 10
batch_idx = 0
batch_no = 0
for num in range(100):
    inq.put(num)
    batch_idx += 1
    if batch_idx == batch:
        inq.join()
        outq.close()
        sum = 0
        for item in outq:
            sum += item
        print(f'Batch {batch_no}, sum = {sum}')
        batch_idx = 0
        batch_no += 1

stop_threads(inq, calc_threads)
