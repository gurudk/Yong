import copy

li1 = [1, 2, [3, 5], 4]
li2 = copy.copy(li1)
print("li1 ID: ", id(li1))
print("li2 ID: ", id(li2), "Value: ", li2)
li3 = copy.deepcopy(li1)
print("li3 ID: ", id(li3), "Value: ", li3)

li1 = [1, 2, [3, 5], 4]
li1_shallow_copy = copy.copy(li1)

li2 = copy.deepcopy(li1)
print("The original elements before deep copying")
for i in range(0, len(li1)):
    print(li1[i], end=" ")

print("\r")
li2[2][0] = 7
print("The new list of elements after deep copying ")
for i in range(0, len(li1)):
    print(li2[i], end=" ")

print("\r")
print("The original elements after deep copying")
for i in range(0, len(li1)):
    print(li1[i], end=" ")

print("===================")
li1[2][0] = 0
print(li1_shallow_copy)
print(li1)
print(li2)
