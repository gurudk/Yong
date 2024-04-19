"""

第32条 考虑用生成器表达式改写数据量较大的列表推导

"""

value = [len(x) for x in open('learning_effective_python_31.py')]
print(value)

it = (len(x) for x in open('learning_effective_python_31.py'))
print(it)
print(next(it))
print(next(it))

roots = ((x, x**0.5) for x in it)
print(roots)
print(next(roots))
print(next(roots))
