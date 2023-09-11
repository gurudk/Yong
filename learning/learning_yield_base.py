
def generate_func(number):
    for x in range(number):
        yield x


g = generate_func(30)
print(next(g))
g.send(3)
print(next(g))