from pathlib import Path
import torch
from melo.api import TTS

torch.cuda.empty_cache()

tr_file_path = "paper/test_tr_output/How To Do Great Work_en_clean.txt"
base_name = Path(tr_file_path).stem
Path("paper/test_wav/" + base_name + "_india").mkdir(parents=True, exist_ok=True)

tr_file = open(tr_file_path, 'r')
lines = tr_file.readlines()
for i in range(0, len(lines)):
    # Speed is adjustable
    speed = 1.0
    device = 'cpu'  # or cuda:0
    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id

    output_path = "paper/test_wav/" + base_name + "_india/" + base_name + "_" + str(i) + ".wav"
    model.tts_to_file(lines[i], speaker_ids['EN_INDIA'], output_path, speed=speed)

    print(f"{i} line tts completed!")
