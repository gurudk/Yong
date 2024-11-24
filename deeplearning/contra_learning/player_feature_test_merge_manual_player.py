import shutil
from pathlib import Path

test_base_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/"
test_base_dir_merge = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/merge/"
merge_base_path = Path(test_base_dir_merge)
for p in Path(test_base_dir).iterdir():
    if p.is_dir():
        sub_dir = str(p).split("/")[-1]
        player_no = sub_dir.split("_")[3]
        player_path = merge_base_path.joinpath(player_no)
        player_path.mkdir(parents=True, exist_ok=True)
        for f in p.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(player_path))
