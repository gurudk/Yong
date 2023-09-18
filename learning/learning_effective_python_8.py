"""

第8条 用zip函数同时遍历两个迭代器

"""
import itertools

names = ['Cecilia', 'Lise', 'Marie']
counts = [len(x) for x in names]

print(counts)

longest_name = None
max_count = 0

for name, count in zip(names, counts):
    if count > max_count:
        longest_name = name
        max_count = count

names.append('Rosalind')

for name ,count in itertools.zip_longest(names, counts, fillvalue='xxx'):
    print(f'{name},{count}')



