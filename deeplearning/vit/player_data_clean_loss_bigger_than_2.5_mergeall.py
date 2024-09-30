import json

input_merge_file = "./player_annotation/clean_body_orientation_atan2_mergeall_07.json.20240928093915"
loss_stat_file = "./explored/player_body_orientation_data_explored_file_140pth_angle.txt.20240929114007"
data_clean_output_file = "./player_annotation/clean_body_orientation_loss25_mergeall_07.json.20240928093915"
data_clean_val_output_file = "./player_annotation/clean_body_orientation_loss15_2nd_mergeall_val_07.json.20240928093915"
untrained_sample_file = "./explored/loss2555555_val_data_explored_file_140pth_angle.txt.20240929155538"

with open(loss_stat_file, 'r') as lf:
    loss_stat_list = json.loads(lf.read())

with open(untrained_sample_file, 'r') as unf:
    untrained_list = json.loads(unf.read())

untrained_list = list(filter(lambda xx: xx[1] < 1.5, untrained_list))

untrained_val_dict = {}
for ue in untrained_list:
    untrained_val_dict[ue[0]] = ue[1]

with open(input_merge_file, 'r') as imf:
    old_mergedata = json.loads(imf.read())

loss25_list = list(filter(lambda xx: xx[1] < 2.5, loss_stat_list))
loss25_dict = {}
for e in loss25_list:
    loss25_dict[e[0]] = e[1]

print(loss25_dict)

clean25_data = {}
clean25_data["file_ids"] = old_mergedata["file_ids"]
player_detections_25 = {}

for key in old_mergedata["player_detections"]:
    if key in loss25_dict:
        player_detections_25[key] = old_mergedata["player_detections"][key]

clean25_data["player_detections"] = player_detections_25

assert 35733 == len(player_detections_25)
print("Good num:", len(player_detections_25))

print("Untrained num:", len(untrained_list))

# with open(data_clean_output_file, 'w') as wf:
#     wf.write(json.dumps(clean25_data))

clean25_data_val = {}
player_detections_val_25 = {}
for key in old_mergedata["player_detections"]:
    if key in untrained_val_dict:
        player_detections_val_25[key] = old_mergedata["player_detections"][key]

clean25_data_val["file_ids"] = old_mergedata["file_ids"]
clean25_data_val["player_detections"] = player_detections_val_25

print(len(clean25_data_val["player_detections"]))

with open(data_clean_val_output_file, 'w') as vf:
    vf.write(json.dumps(clean25_data_val))

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
