from pydub import AudioSegment
from pathlib import Path

import re

sounds = []

file_path = "paulgraham/wav_files/cities"
paper_parent = Path(file_path).parent.resolve().parent
suffix_name = file_path.split("/")[-1]
out_path = paper_parent.joinpath("merged_mp3").joinpath(suffix_name + ".mp3")

from os import listdir
from os.path import isfile, join

onlyfiles = [join(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]

onlyfiles.sort(key=lambda name: int(re.split(r"[_|\.]+", name)[-2]))

for f in onlyfiles:
    sounds.append(AudioSegment.from_wav(f))

playlist = AudioSegment.empty()
for i in range(0, len(onlyfiles)):
    playlist += sounds[i]
    print("sounds " + str(i) + " added!")

print("===================================================")
print("merge started!")
playlist.export(out_path, format="mp3")
print("merge completed!")
