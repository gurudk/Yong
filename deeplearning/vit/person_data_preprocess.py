import json

input_file = "./player_annotation/player_annotated_val_07.json.20240925120326"
out_file = "./player_annotation/onlyperson_val_07.json.20240925120326"

with open(input_file, 'r') as rf:
    input_dict = json.loads(rf.read())

data_clean = {}
file_ids = {}
player_detections = {}
for key in input_dict["player_detections"]:
    if "sports ball" not in key:
        player_detections[key] = input_dict["player_detections"][key]

data_clean["file_ids"] = input_dict["file_ids"]
data_clean["player_detections"] = player_detections

with open(out_file, 'w') as wf:
    wf.write(json.dumps(data_clean))
