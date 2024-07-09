from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu'  # or cuda:0

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
test_text = "2023年至今，Deep Learning 正在飞速发展，在NLP领域，以ChatGPT为代表的LLM模型对长文本的理解有了巨大的提升，使得AI具有了深度思考的能力。AIGC领域也迎来了爆发，诞生了处理Text-to-Image、Image-to-Image任务的Stable-Difusion、SDXL、ControlNet模型，以及文生视频的Sora模型。"
text_1 = '说中文, ni3 hao3吗, BBC广播, CNN新闻, 美国HBO, NBC转播, CCTV一套, HIV病毒, 100%不合格, 列宁格勒，希特勒，P2P软件, Sony的PSP, 微软的XBOX游戏机, 2018年初，1931年, 轰炸地堡，数了数几只羊'
text_2 = "一年好景君须记，正是橙黄橘绿时。 解析： 此诗是苏轼于宋哲宗元佑五年（1090）任杭州太守时所作。 刘景文名刘季孙，是苏轼的密友。 这首诗的意味非常丰富。 但在苏轼婉约、豪放并行的词风下，却当属其温润含蓄的一面。 此诗前半首说“荷尽菊残”都是为了强调荷莲走到了生命尽头，已经再无花中豪杰那种“接天莲叶无穷碧，映日荷花别样红”的盛景，也无霜叶一般“霜叶红于二月花”的雅致了"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh_test_2.wav'
model.tts_to_file(text_2, speaker_ids['ZH'], output_path, speed=speed)
