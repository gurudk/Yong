import os
import pathlib

import json
import re
from pathlib import Path
import shutil


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


def sample_players(player_json, frame_span):
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


def sort_players(player_json):
    new_json = {}
    for key in player_json.keys():
        temp_list = player_json[key]
        temp_list = sorted(temp_list, key=lambda x: (x.video_no, x.frame_no))
        new_json[key] = temp_list
    return new_json


def stat_players(player_json):
    for key in player_json.keys():
        print(key, ":", len(player_json[key]))


def gen_player_dataset(player_dataset_dir):
    player_json = {}
    for p in Path(player_dataset_dir).iterdir():
        if p.is_dir():
            sub_dir_name = p.stem

            strs = sub_dir_name.split("_")
            if len(strs) >= 4:
                match_no = strs[0]
                video_no = strs[1]
                track_id = strs[2]

                if len(strs) == 5 and re.search(r"^[a-zA-Z][\d]+$", strs[3]) and strs[4] == "player":
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


def dump_player_dataset(player_json, out_dir):
    out_dir_path = Path(out_dir)
    for key in player_json.keys():
        sub_dir_path = out_dir_path.joinpath(key)
        sub_dir_path.mkdir(parents=True, exist_ok=True)
        for pl in player_json[key]:
            src_file_name = pl.file_path.split("/")[-1]
            out_file_path = sub_dir_path.joinpath(src_file_name)
            shutil.copy2(pl.file_path, str(sub_dir_path))
            # with open(pl.file_path, 'r') as rf:
            #     with open(str(out_file_path), 'w') as wf:
            #         wf.write(rf.read())


def count_player_samples(player_json):
    count_sample = 0
    for key in player_json.keys():
        count_sample += len(player_json[key])

    return count_sample


player_dataset_dir0 = "/home/wolf/datasets/reid/DFL/SR583_0_manual_clean/"
player_dataset_dir1 = "/home/wolf/datasets/reid/DFL/SR583_1_manual_clean/"
player_dataset_dir2 = "/home/wolf/datasets/reid/DFL/SR42_0_manual_clean/"
player_frame_span = 5

merge_ds = merge_players(gen_player_dataset(player_dataset_dir0), gen_player_dataset(player_dataset_dir1))
ds2 = gen_player_dataset(player_dataset_dir2)

merge_ds = merge_players(merge_ds, ds2)

sorted_json = sort_players(merge_ds)
sampled_json = sample_players(sorted_json, frame_span=player_frame_span)

stat_players(sorted_json)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
stat_players(sampled_json)

sample5_dump_out_dir = "/home/wolf/datasets/reid/dataset/test_gen_SR583_SR42_sampled5/"
merged_dump_out_dir = "/home/wolf/datasets/reid/dataset/test_gen_SR583_SR42_merged/"
dump_player_dataset(sampled_json, sample5_dump_out_dir)
dump_player_dataset(merge_ds, merged_dump_out_dir)
# dump_player_dataset(sampled_json, "/home/wolf/datasets/reid/dataset/test_gen_SR583_SR42_sampled8")

print("Total  num:", count_player_samples(merge_ds))
print("Sample  num:", count_player_samples(sampled_json))

not_team_count = 0
for ky in sampled_json.keys():
    if "team" not in ky:
        not_team_count += len(sampled_json[ky])

print("not team player count:", not_team_count)

team_count = 0
for ky in sampled_json.keys():
    if "team_b_" in ky:
        team_count += len(sampled_json[ky])

print("team b player count:", team_count)

team_count = 0
for ky in sampled_json.keys():
    if "team_w_" in ky:
        team_count += len(sampled_json[ky])

print("team w player count:", team_count)
