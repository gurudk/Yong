from pydub import AudioSegment

# sound1 = AudioSegment.from_file("paper/test_wav/tei_test_simple/tei_test_simple_0.wav", format="wav")
# sound2 = AudioSegment.from_file("paper/test_wav/tei_test_simple/tei_test_simple_1.wav", format="wav")
#
# # sound1 6 dB louder
# # louder = sound1 + 6
#
#
# # sound1, with sound2 appended (use louder instead of sound1 to append the louder version)
# combined = sound1 + sound2
#
# # simple export
# file_handle = combined.export("test_merge.mp3", format="mp3")

import re

sounds = []

file_path = "paper/test_wav/How To Do Great Work_en_clean_us"

from os import listdir
from os.path import isfile, join

onlyfiles = [join(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]

onlyfiles.sort(key=lambda name: int(re.split(r"[_|\.]+", name)[-2]))

for f in onlyfiles:
    sounds.append(AudioSegment.from_wav(f))

playlist = AudioSegment.empty()
for sound in sounds:
    playlist += sound

playlist.export("How To Do Great Work_en_us.mp3", format="mp3")
