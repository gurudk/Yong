work_dir = "/home/wolf/papers_backup"

from pathlib import Path
import os

work_path = Path(work_dir)

for f in work_path.rglob("*.pdf"):
    # print(f)
    file_name = f.name
    if ":" in str(f):
        new_name = file_name.replace(":", "_")
        print(new_name)
        new_path = f.parent.joinpath(new_name)
        print(new_path)
        f.rename(new_path)
