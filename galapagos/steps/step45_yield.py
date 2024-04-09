def ye():
    for i in range(10):
        yield i


def ye_from():
    yield from ye()


for d in ye_from():
    print(d)

a = [1, 2, 3, 4, 5]
print(a[:-1])
