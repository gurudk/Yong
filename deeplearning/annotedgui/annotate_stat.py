import glob
import json
import os

# Use the glob module to find all files in the current directory with a ".txt" extension.
files = glob.glob("./annotated/*.*.*.*")

# Sort the list of file names based on the modification time (getmtime) of each file.
files.sort(key=os.path.getmtime)
all_images = {}
for file_name in files:
    with open(file_name, 'r') as fi:
        dic = json.loads(fi.read())
        for key in dic:
            all_images[key] = dic[key]

print(len(all_images))
