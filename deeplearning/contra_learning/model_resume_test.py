import os
import glob
import re


def get_latest_model_file(model_dir):
    files = list(filter(os.path.isfile, glob.glob(model_dir + "*")))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    latest_epoch = int(re.split(r"[_\.]", latest_file)[-2])

    return latest_file, latest_epoch


print(get_latest_model_file("./zoo/"))
