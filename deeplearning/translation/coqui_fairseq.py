import torch
from TTS.api import TTS

torch.cuda.empty_cache()

# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# List available 🐸TTS models
# print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(
    text="一年好景君须记，正是橙黄橘绿时。 解析： 此诗是苏轼于宋哲宗元佑五年（1090）任杭州太守时所作。 刘景文名刘季孙，是苏轼的密友。 这首诗的意味非常丰富。 但在苏轼婉约、豪放并行的词风下，却当属其温润含蓄的一面。 此诗前半首说“荷尽菊残”都是为了强调荷莲走到了生命尽头，已经再无花中豪杰那种“接天莲叶无穷碧，映日荷花别样红”的盛景，也无霜叶一般“霜叶红于二月花”的雅致了",
    file_path="output.wav")

# api = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v1.1").to("cuda")
# api.tts_to_file(text="This is a test.", file_path="output.wav")
