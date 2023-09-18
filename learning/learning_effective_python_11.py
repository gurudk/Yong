"""

第11条 学会对序列作切片
凡是实现了__getitem__和__setitem__的类都可以切割

"""

a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
print('Middle two:', a[3:5])
print('All but ends:', a[::-1])

print(a[2:3])
print(a[2])
assert a[2:3] == [a[2]]
print(a[2:-1])
print(a[-3:-1])

print('Before:',a)
a[2:3] = [47, 11]
print('After:', a)

b = a[:]

assert b == a and b is not a

b = a
print('Before a:', a)
print('Before b:', b)

a[:] = [101, 102, 103]  # 全部替换掉
assert a == b
print('After a:', a)
print('After b:', b)


