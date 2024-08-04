import json

jlist = []
d1 = {}
d1['/home/1.jpg'] = "100,200,300,400"
jlist.append(d1)
jlist.append({"lksfdj": "lsdjflds"})
d1['/home/2.png'] = "1,2,3,4"

print(json.dumps(d1))

d2 = json.loads(json.dumps(d1))
d2["/home/3.jpg"] = "123,234,333,444"
print(d2)
