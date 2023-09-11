import timeit


def child():
    for i in range(1_000_000):
        yield i


def slow():
    for i in child():
        yield i


def fast():
    yield from child()


baseline = timeit.timeit(stmt='for _ in slow():pass', globals= globals(), number=50)
print(f'Manual nesting {baseline:.2f}s')
comparison = timeit.timeit(stmt='for _ in fast():pass', globals=globals(), number=50)
print(f'Composed nesting {comparison:.2f}s')
reduction = -(comparison - baseline)/baseline
print(f'{reduction:.1%}s less time')


import_module = "import random"
testcode = ''' 
def test(): 
    return random.(10, 100)randint
'''
print(timeit.repeat(stmt=testcode, setup=import_module))
