"""
第27条 用列表推导替代map和filter
"""

a = [*range(1, 11, 1)]
print(a)

squares = [x**2 for x in a]
print(squares)

even_squares = [x**2 for x in a if x % 2 == 0]
print(even_squares)

alt = map(lambda x: x**2, a)
print(list(alt))

even_squares_dict = {x: x**2 for x in a}
print(even_squares_dict)