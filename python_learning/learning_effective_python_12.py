"""

第12条 不要再切片里同时指定起止下标和步进

"""

x = ['red', 'orange', 'yellow', 'green', 'blue','purple']
odds = x[::2]
evens = x[1::2]
print(odds)
print(evens)

x = b'mongoose'
y = x[::-1]
print(y)

x = "电脑"
y = x[::-1]
print(y)

x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
print(x[::2])
print(x[::-2])  # 反方向，隔两个选一个
print(x[-2:2:-2])  # ['g', 'e']



