import os.path
import time

import requests, uuid, json
import re
import random

from xml.dom import minidom
from pathlib import Path

from melo.api import TTS


def generate_wav_file(response, output_path):
    # Speed is adjustable
    speed = 1.0
    device = 'cpu'  # or cuda:0

    # text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
    # test_text = "2023年至今，Deep Learning 正在飞速发展，在NLP领域，以ChatGPT为代表的LLM模型对长文本的理解有了巨大的提升，使得AI具有了深度思考的能力。AIGC领域也迎来了爆发，诞生了处理Text-to-Image、Image-to-Image任务的Stable-Difusion、SDXL、ControlNet模型，以及文生视频的Sora模型。"
    # text_1 = '说中文, ni3 hao3吗, BBC广播, CNN新闻, 美国HBO, NBC转播, CCTV一套, HIV病毒, 100%不合格, 列宁格勒，希特勒，P2P软件, Sony的PSP, 微软的XBOX游戏机, 2018年初，1931年, 轰炸地堡，数了数几只羊'
    ms_text = response[0]["translations"][0]["text"]

    model = TTS(language='ZH', device=device)
    speaker_ids = model.hps.data.spk2id

    # output_path = 'zh_ms_test.wav'
    model.tts_to_file(ms_text, speaker_ids['ZH'], output_path, speed=speed)


def get_content(file_name):
    str_content = []
    # file = open(file_name, 'r')
    # print(file.read())
    doc = minidom.parse(file_name)
    header = doc.getElementsByTagName("teiHeader")[0]
    # print(header.firstChild.data)

    title = header.getElementsByTagName("title")[0]

    print(title.firstChild.data)
    str_content.append(title.firstChild.data)

    abstract = header.getElementsByTagName("abstract")[0]

    for div in abstract.getElementsByTagName("div"):
        for s in div.getElementsByTagName("s"):
            str_seg = ""
            nodes = s.childNodes
            for node in nodes:
                if node.nodeName == "ref":
                    str_seg += node.firstChild.data + " "
                else:
                    str_seg += node.data + " "
            # str = str.replace("\n", " ")
            str_seg = re.sub(r"[\s][\s]+", " ", str_seg)
            print(str_seg)
            str_content.append(str_seg)

    body = doc.getElementsByTagName("body")[0]

    divs = body.getElementsByTagName("div")

    for div in divs:
        str_head = ""
        head = div.getElementsByTagName("head")
        if head:
            chapter = head[0].getAttribute("n")
            if chapter:
                str_head += chapter + " "
            str_head += head[0].firstChild.data
        print(str_head)
        str_content.append(str_head)
        ss = div.getElementsByTagName("s")
        for s in ss:
            str_seg = ""
            nodes = s.childNodes
            for node in nodes:
                if node.nodeName == "ref":
                    str_seg += node.firstChild.data + " "
                else:
                    str_seg += node.data + " "
            # str = str.replace("\n", " ")
            str_seg = re.sub(r"[\s][\s]+", " ", str_seg)

            str_content.append(str_seg)
            print(str_seg)

    return str_content


# Add your key and endpoint
key = os.getenv("ms_api_key")
endpoint = "https://api.cognitive.microsofttranslator.com"

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "eastasia"

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'en',
    'to': ['zh-Hans']
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

file_path = "paper/test_tei/tei_test_simple.xml"
base_name = Path(file_path).stem
out_txt_file = "paper/test_tr_output/" + base_name + ".txt"

contents = get_content(file_path)
tr_results = []
for i in range(0, len(contents)):
    # You can pass more than one object in body.
    body = [{
        'text': contents[i]
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    tr_results.append(response[0]["translations"][0]["text"])
    print(response)
    time.sleep(random.uniform(0.1, 0.5))

# for content in contents:
#     # You can pass more than one object in body.
#     body = [{
#         'text': content
#     }]
#
#     request = requests.post(constructed_url, params=params, headers=headers, json=body)
#     response = request.json()
#
#     tr_results.append(response[0]["translations"][0]["text"])
#     print(response)
#     time.sleep(random.uniform(0.1, 0.5))

# write translation result to file
out_txt_file = open(out_txt_file, 'w')
for line in tr_results:
    out_txt_file.write(line + '\n')

out_txt_file.close()
print("Translation completed!")
