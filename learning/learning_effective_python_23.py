"""
第23条 用关键字参数表示可选的行为
"""


def remainder(number, divisor):
    return number % divisor


remainder(20, 7)
remainder(20, divisor=7)
remainder(number=20, divisor=7)
remainder(divisor=7, number=20)

my_kwargs = {
    "number": 20,
    "divisor": 7
}


assert remainder(**my_kwargs) == 6

my_kwargs_1 = {
    "number": 20
}

my_kwargs_2 = {
    "divisor": 7
}

assert remainder(**my_kwargs_1, **my_kwargs_2) == 6

print("OK")