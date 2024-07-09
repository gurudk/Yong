from pathlib import Path
import torch
from melo.api import TTS
import re

torch.cuda.empty_cache()

tr_file_path = "paper/test_tr_output/A brief survey for visual saliency detection_cn.txt"


def generate_cn_audio(file_path):
    base_name = Path(tr_file_path).stem
    paper_parent = Path(file_path).parent.resolve().parent
    out_dir_path = paper_parent.joinpath("test_wav").joinpath(base_name)

    out_dir_path.mkdir(parents=True, exist_ok=True)

    tr_file = open(tr_file_path, 'r')
    lines = tr_file.readlines()
    for i in range(236, len(lines)):
        str_line = re.sub(r"[\s]+", "", lines[i])
        if str_line != "":
            # Speed is adjustable
            speed = 1.0
            device = 'cpu'  # or cuda:0
            model = TTS(language='ZH', device=device)
            speaker_ids = model.hps.data.spk2id

            output_path = out_dir_path.joinpath(base_name + "_" + str(i) + ".wav")
            model.tts_to_file(lines[i], speaker_ids['ZH'], output_path, speed=speed)

            print(f"{i} line tts completed!")


generate_cn_audio(tr_file_path)
