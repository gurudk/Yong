import json
import shutil
import datetime
from pathlib import Path


def get_nowtime_str():
    nowtime = datetime.datetime.now()

    return nowtime.strftime("%Y%m%d%H%M%S")


def dump_json_dataset_to_disk(classify_json_file, dump_dir):
    with open(classify_json_file, 'r') as rf:
        dump_path = Path(dump_dir)

        player_json = json.loads(rf.read())
        print(player_json["player_indexes"])
        player_list = player_json["player_list"]
        player_indexes = player_json["player_indexes"]
        for idx, file_path in enumerate(player_list.keys()):
            player_category = player_indexes[str(player_list[file_path])]
            sub_dir = dump_path.joinpath(player_category)
            sub_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, str(sub_dir))

            if idx % 500 == 0 and idx != 0:
                print(idx, "players is dumped~")


classify_json_file = "/home/wolf/datasets/reid/dataset/classify/player_classify_final_span5.json.20241119102915"
dump_out_dir = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_" + get_nowtime_str()

dump_json_dataset_to_disk(classify_json_file, dump_out_dir)
