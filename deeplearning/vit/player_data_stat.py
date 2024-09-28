import json
from PIL import Image

train_file = "./player_annotation/clean_body_orientation_atan2_val_07.json.20240928100332"

total_width = 0.0
total_height = 0.0

with open(train_file, 'r') as rf:
    json_dict = json.loads(rf.read())
    print(len(json_dict["player_detections"]))

    count = len(json_dict["player_detections"])
    for key in json_dict["player_detections"]:
        img = Image.open(key)
        total_height += img.height
        total_width += img.width

    print(total_width / count, total_height / count)
