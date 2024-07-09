from pydub import AudioSegment
from pathlib import Path

import re

sounds = []

file_path = "paper/test_wav/Let Go of Your Labels with Unsupervised Transfer_cn"
paper_parent = Path(file_path).parent.resolve().parent
suffix_name = file_path.split("/")[-1]
out_path = paper_parent.joinpath("test_merge").joinpath(suffix_name + ".mp3")

from os import listdir
from os.path import isfile, join

onlyfiles = [join(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]

onlyfiles.sort(key=lambda name: int(re.split(r"[_|\.]+", name)[-2]))

for f in onlyfiles:
    sounds.append(AudioSegment.from_wav(f))

playlist = AudioSegment.empty()
for sound in sounds:
    playlist += sound

playlist.export(out_path, format="mp3")
