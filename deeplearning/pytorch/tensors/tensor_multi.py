x = [256] * 3 * 3
print(x)

h = [256] * 2
y = zip([256] + h, h + [4])
print([256] + h)
print(h + [4])

for n, k in y:
    print(n, k)
