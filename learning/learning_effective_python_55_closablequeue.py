
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
        for item in self.in_queue:
            result = self.func(item)
            self.out_queue.put(result)


def download(item):
    ...


def resize(item):
    ...


def upload(item):
    ...


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


download_queue = ClosableQueue()
resize_queue = ClosableQueue()
upload_queue = ClosableQueue()
done_queue = ClosableQueue()

# threads = [
#     StoppableWorker(download, download_queue, resize_queue),
#     StoppableWorker(resize, resize_queue, upload_queue),
#     StoppableWorker(upload, upload_queue, done_queue)
# ]


# for thread in threads:
#     thread.start()
#
# for _ in range(1000):
#     download_queue.put(object())
#
# download_queue.close()
# download_queue.join()
#
# resize_queue.close()
# resize_queue.join()
#
# upload_queue.close()
# upload_queue.join()  # wait all task_done
# print(done_queue.qsize(), ' item finished')
#
# for thread in threads:
#     thread.join()


download_threads = start_threads(3, download, download_queue, resize_queue)
resize_threads = start_threads(4, resize, resize_queue, upload_queue)
upload_threads = start_threads(5, upload, upload_queue, done_queue)

for _ in range(100):
    download_queue.put(object())

stop_threads(download_queue, download_threads)
stop_threads(resize_queue, resize_threads)
stop_threads(upload_queue, upload_threads)

print(done_queue.qsize(), ' items finished')

