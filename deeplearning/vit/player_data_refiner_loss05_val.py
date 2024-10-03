import json

input_merge_file = "./player_annotation/clean_body_orientation_atan2_mergeall_07.json.20240928093915"
train_filter_file = "./explored/new_explore_140th.txt.20241003090118"
data_clean_output_file = "./player_annotation/new_data_loss05_mergeall_07.json.20240928093915"
data_clean_val_output_file = "./player_annotation/new_data_loss05_mergeall_val_data_07.json.20240928093915"
untrained_sample_file = "./explored/new_data_loss05_valdata.txt.20241003123727"

with open(train_filter_file, 'r') as lf:
    loss_stat_list = json.loads(lf.read())

with open(untrained_sample_file, 'r') as unf:
    untrained_list = json.loads(unf.read())

untrained_list = list(filter(lambda xx: xx[1] < 0.5, untrained_list))

untrained_val_dict = {}
for ue in untrained_list:
    untrained_val_dict[ue[0]] = ue[1]

with open(input_merge_file, 'r') as imf:
    old_mergedata = json.loads(imf.read())

loss05_list = list(filter(lambda xx: xx[1] < 0.5, loss_stat_list))
loss05_dict = {}
for e in loss05_list:
    loss05_dict[e[0]] = e[1]

print(loss05_dict)

clean05_data = {}
clean05_data["file_ids"] = old_mergedata["file_ids"]
player_detections_05 = {}

for key in old_mergedata["player_detections"]:
    if key in loss05_dict:
        player_detections_05[key] = old_mergedata["player_detections"][key]

clean05_data["player_detections"] = player_detections_05

assert 31283 == len(player_detections_05)
print("Good num:", len(player_detections_05))

print("Untrained num:", len(untrained_list))

with open(data_clean_output_file, 'w') as wf:
    wf.write(json.dumps(clean05_data))

clean05_data_val = {}
player_detections_val_05 = {}
for key in old_mergedata["player_detections"]:
    if key in untrained_val_dict:
        player_detections_val_05[key] = old_mergedata["player_detections"][key]

clean05_data_val["file_ids"] = old_mergedata["file_ids"]
clean05_data_val["player_detections"] = player_detections_val_05

print(len(clean05_data_val["player_detections"]))

assert 3167 == len(player_detections_val_05)
with open(data_clean_val_output_file, 'w') as vf:
    vf.write(json.dumps(clean05_data_val))

# found_replicated_sample = False
# for key in loss25_dict:
#     if key in untrained_val_dict:
#         print("replicated sample found in train and val dataset~")
#         found_replicated_sample = True
#
# if found_replicated_sample:
#     print("Replicated sample found!!!")
# else:
#     print("No replicated sample in train and val dataset, good news!")
