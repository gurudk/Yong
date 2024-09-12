import os
import json

annotated_file = "./released/annotated.release.20240912113708.txt"
train_dir = "/home/wolf/datasets/final_dataset_evo_v3/train/images"

json_obj = {}
with open(annotated_file, 'r') as f:
    json_obj = json.loads(f.read())

for root, dirs, files in os.walk(train_dir):
    for file_name in files:
        file_path = root + "/" + file_name
        if file_path not in json_obj:
            print(file_path)
