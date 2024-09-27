import json

from datetime import datetime


def get_nowtime():
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


dataset_file = "./player_annotation/player_annotated.json.20240924174013"
output_file = "./player_annotation/split_3000_player.json." + get_nowtime()

with open(dataset_file, 'r') as rf:
    dataset_json = json.loads(rf.read())

image_num = 300

subdataset = {}
file_ids = {}
player_detections = {}

for key in dataset_json["file_ids"]:
    if int(key) < image_num:
        file_ids[key] = dataset_json["file_ids"][key]

for key in dataset_json["player_detections"]:
    image_index = dataset_json["player_detections"][key]["from_image"]

    if int(image_index) < image_num:
        player_detections[key] = dataset_json["player_detections"][key]

subdataset["file_ids"] = file_ids
subdataset["player_detections"] = player_detections

print(len(subdataset["player_detections"]))

with open(output_file, 'w') as wf:
    wf.write(json.dumps(subdataset))
