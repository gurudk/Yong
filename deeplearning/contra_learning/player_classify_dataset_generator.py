import os
import re
import json
import datetime

from pathlib import Path


class Player:
    def __init__(self, player_key, match_no, video_no, track_id, frame_no, team_no, player_no, player_conf, file_path):
        assert video_no

        self.player_key = player_key
        self.match_no = match_no
        self.video_no = int(video_no)
        self.track_id = track_id
        self.frame_no = int(frame_no)
        self.team_no = team_no
        self.player_no = player_no
        self.player_conf = player_conf
        self.file_path = file_path


def gen_video_dataset(player_dataset_dir):
    player_json = {}
    for p in Path(player_dataset_dir).iterdir():
        if p.is_dir():
            sub_dir_name = p.stem

            strs = sub_dir_name.split("_")
            if len(strs) >= 4:
                match_no = strs[0]
                video_no = strs[1]
                track_id = strs[2]

                if len(strs) >= 5 and re.search(r"^[a-zA-Z][\d]+$", strs[3]) and strs[4] == "player":
                    team_no = strs[3][0:1].lower()
                    player_no = strs[3][1:]
                    player_key = match_no + "_team_" + team_no.lower() + "_" + strs[3].lower()
                else:
                    team_no = "na"
                    player_no = "na"
                    player_key = match_no + "_" + strs[3].lower()

                for pfile in p.iterdir():
                    if pfile.is_file():
                        pfile_stem = pfile.stem
                        ss = pfile_stem.split("_")
                        player_frame_no = ss[4]
                        player_conf = ss[5]
                        player = Player(player_key, match_no, video_no, track_id, player_frame_no, team_no, player_no,
                                        player_conf, str(pfile))

                        if player_json.get(player_key):
                            player_list = player_json.get(player_key)
                            player_list.append(player)
                        else:
                            player_json[player_key] = [player]

            # print(match_no, video_no, track_id, player_key, team_no, player_no)
    return player_json


def merge_players(player_json1, player_json2):
    merged_json = {}
    for key1 in player_json1.keys():
        if key1 not in player_json2.keys():
            # only json1
            merged_json[key1] = player_json1[key1]
        else:
            # merge same key in json1 and json2
            merged_json[key1] = player_json1[key1] + player_json2[key1]

    for key2 in player_json2.keys():
        if key2 not in player_json1.keys():
            merged_json[key2] = player_json2[key2]

    return merged_json


def sort_players(player_json):
    new_json = {}
    for key in player_json.keys():
        temp_list = player_json[key]
        temp_list = sorted(temp_list, key=lambda x: (x.video_no, x.frame_no))
        new_json[key] = temp_list
    return new_json


def sample_players(player_json, frame_span=5):
    new_json = {}
    for key in player_json.keys():
        player_list = player_json[key]
        new_list = []
        if player_list:
            cur_sample_player = player_list[0]
            new_list.append(player_list[0])
            for pl in player_list:
                if pl.track_id == cur_sample_player.track_id and pl.frame_no - cur_sample_player.frame_no < frame_span:
                    continue
                else:
                    new_list.append(pl)
                    cur_sample_player = pl
        new_json[key] = new_list

    return new_json


def gen_player_dataset(base_dir_input):
    ret_player_json = {}
    for part in Path(base_dir_input).iterdir():
        if part.is_dir():
            for sr_dir in part.iterdir():
                if sr_dir.is_dir():
                    if str(sr_dir).endswith("_ready"):
                        video_json = gen_video_dataset(str(sr_dir))
                        ret_player_json = merge_players(ret_player_json, video_json)

    return ret_player_json


def get_nowtime_str():
    nowtime = datetime.datetime.now()

    return nowtime.strftime("%Y%m%d%H%M%S")


test_sr_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR26/SR26_1_manual_ready"
base_dir = "/home/wolf/datasets/reid/DFL/dest_manual/"
base_path = Path(base_dir)
dump_json_file = "/home/wolf/datasets/reid/dataset/classify/player_classify_span5.json." + get_nowtime_str()

# players_json = {}
# for part in base_path.iterdir():
#     if part.is_dir():
#         for sr_dir in part.iterdir():
#             if sr_dir.is_dir():
#                 if str(sr_dir).endswith("_ready"):
#                     print(sr_dir)

# players_json = gen_video_dataset(test_sr_dir)
# sample_player_json = sample_players(sort_players(players_json), frame_span=5)
# for key in players_json.keys():
#     print(key, len(players_json[key]))
#
# print("=================================sample 5 summary=============================")
#
# for key in sample_player_json.keys():
#     print(key, len(sample_player_json[key]))
#
# print("=================================details =============================")

# for key in players_json.keys():
#     if key == "SR26_team_b_b26":
#         for p in players_json[key]:
#             print(p.file_path)


players_json = gen_player_dataset(base_dir)
sample5_json = sample_players(sort_players(players_json), frame_span=5)

player_sum = 0
for key in sample5_json.keys():
    player_sum += len(sample5_json[key])
    print(key, len(sample5_json[key]))

print("Samples sum:", player_sum)
print(sorted(sample5_json.keys()))
print()
print()
print("================================= gen dataset details =============================")
player_dicts = {}
sort_categories = sorted((sample5_json.keys()))
for idx, player in enumerate(sort_categories):
    player_dicts[player] = idx
print(player_dicts)

player_final_json = {}
for key in sample5_json.keys():
    for player in sample5_json[key]:
        player_final_json[player.file_path] = player_dicts[key]

print(len(player_final_json))
player_dataset = {"player_dict": player_dicts, "player_list": player_final_json}

with open(dump_json_file, 'w') as wf:
    wf.write(json.dumps(player_dataset))

# reslut_json_file = "/home/wolf/datasets/reid/dataset/classify/player_classify_span3.json.20241107125947"
#
# with open(reslut_json_file, 'r') as rf:
#     json_dump_player = json.loads(rf.read())
#
#     for key in json_dump_player.keys():
#         print(key, ":", json_dump_player[key])
#     print(len(json_dump_player))
