import re
import numpy as np
import matplotlib.pyplot as plt

train_log = "./log/train.log.20240929160418"

log_list = []
with open(train_log, 'r') as lf:
    line = lf.readline()
    while line:
        arr = re.split(r"[\[\]\:\,\/]", line)
        log_list.append((arr[1], arr[2], arr[4], arr[6]))
        line = lf.readline()

nplist = np.array(log_list)

print(nplist[120:200, 3])
fg, ax = plt.subplots()
ax.plot(np.int32(nplist[0:220, 0]), np.float32(nplist[0:220, 3]), label='val')
ax.plot(np.int32(nplist[0:220, 0]), np.float32(nplist[0:220, 2]), label='train')
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.xlabel('Iterations')
plt.ylabel('loss ')
plt.title("Traning and Validation Curve")
plt.legend()
plt.show()
