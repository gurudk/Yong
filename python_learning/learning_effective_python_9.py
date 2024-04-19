"""

第9条 不要在for和while循环后使用else块

"""

for i in range(3):
    print('Loop', i)
else:
    print('for cycle is successful, so exec Else block')


for i in range(3):
    print('Loop', i)
    if i == 1:
        break
else:
    print('for cycle is not successful, so not exec Else block')