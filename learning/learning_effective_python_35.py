class MyError(Exception):
    pass


def my_generator():
    yield 1
    try:
        yield 2
    except MyError:
        print('got error')
    else:
        yield 3
    yield 4


it = my_generator()
print(next(it))
print(next(it))
except_out = it.throw(MyError('test'))
print(except_out)


def announce(remaining):
    print(f"{remaining} ticks remaining")


def check_for_reset():
    return False


class Timer:
    def __init__(self,period):
        self.current = period
        self.period = period

    def reset(self):
        self.current = self.period

    def __iter__(self):
        while self.current:
            self.current -= 1
            yield self.current


def run():
    timer = Timer(4)
    for current in timer:
        if check_for_reset():
            timer.reset()
        announce(current)


run()

