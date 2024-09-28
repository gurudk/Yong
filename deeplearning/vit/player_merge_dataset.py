import json

file1 = "./player_annotation/clean_body_orientation_atan2_07.json.20240928093915"
file2 = "./player_annotation/clean_body_orientation_atan2_val_07.json.20240928100332"
out_file = "./player_annotation/clean_body_orientation_atan2_mergeall_07.json.20240928093915"

merge_file_ids = {}
merge_player_detections = {}
merge_data = {}
with open(file1, 'r') as f1:
    with open(file2, 'r') as f2:
        json1 = json.loads(f1.read())
        json2 = json.loads(f2.read())
        print(len(json1["player_detections"]))
        print(len(json2["player_detections"]))
        detr_num1 = len(json1["player_detections"])
        detr_num2 = len(json2["player_detections"])
        sample1_img_num = len(json1["file_ids"])
        sample2_img_num = len(json2["file_ids"])
        print(sample1_img_num, sample2_img_num)
        for key in json1["file_ids"]:
            merge_file_ids[key] = json1["file_ids"][key]

        for key in json2["file_ids"]:
            merge_file_ids[sample1_img_num + int(key)] = json2["file_ids"][key]

        for key in json1["player_detections"]:
            merge_player_detections[key] = json1["player_detections"][key]

        for key in json2["player_detections"]:
            merge_player_detections[key] = json2["player_detections"][key]
            merge_player_detections[key]["from_image"] = detr_num1 + int(json2["player_detections"][key]["from_image"])

print(len(merge_file_ids))
print(len(merge_player_detections))

merge_data["file_ids"] = merge_file_ids
merge_data["player_detections"] = merge_player_detections

with open(out_file, 'w') as wf:
    wf.write(json.dumps(merge_data))
