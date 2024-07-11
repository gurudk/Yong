text = "The problem is the receptor it binds  to:. dressing up is inevitably a substitute for good ideas."

from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto'  # Will automatically use GPU if available

model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# British accent
output_path = 'en-br.wav'
model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)
