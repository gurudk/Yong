import glob
import os
import json

# Use the glob module to find all files in the current directory with a ".txt" extension.
files = glob.glob("./annotated/*.*")

# Sort the list of file names based on the modification time (getmtime) of each file.
files.sort(key=os.path.getmtime)

print("\n".join(files))

print(files[-1])

with open(files[-1], 'r') as f:
    obj = json.loads(f.read())
    print(obj.keys())
    print(list(obj.keys())[-1])
