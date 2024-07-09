from pathlib import Path
import torch
from melo.api import TTS

torch.cuda.empty_cache()

tr_file_path = "paper/test_tr_output/tei_test_simple.txt"
base_name = Path(tr_file_path).stem
Path("paper/test_wav/" + base_name).mkdir(parents=True, exist_ok=True)

tr_file = open(tr_file_path, 'r')
lines = tr_file.readlines()
for i in range(159, 161):
    # Speed is adjustable
    speed = 1.0
    device = 'cpu'  # or cuda:0
    model = TTS(language='ZH', device=device)
    speaker_ids = model.hps.data.spk2id

    output_path = "paper/test_wav/" + base_name + "/" + base_name + "_" + str(i) + ".wav"
    model.tts_to_file(lines[i], speaker_ids['ZH'], output_path, speed=speed)

    print("{i} line tts completed!")
