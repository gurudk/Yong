import os.path
import time

import requests, uuid, json
import re
import random
import os

from xml.dom import minidom
from pathlib import Path
from ai_extract_contents import save_contents


def get_source_text(file_path):
    infile = open(file_path, 'r')
    src_lines = []
    line = infile.readline()
    while line != "":
        src_lines.append(line)
        line = infile.readline()

    infile.close()
    return src_lines


# Add your key and endpoint
key = os.environ["ms_api_code"]
# key = "093fd9512d994764beb3dbfebff5c55f"
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

file_path = "paper/test_tr_output/On the Opportunities and Risks of Foundation Models.txt"
base_name = Path(file_path).stem
out_txt_file = "paper/test_tr_output/" + base_name + "_cn.txt"

contents = get_source_text(file_path)
# print(contents)
# print(len(contents))

tr_results = []
for i in range(0, len(contents)):
    # You can pass more than one object in body.
    body = [{
        'text': contents[i]
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    print(response)
    tr_results.append(response[0]["translations"][0]["text"])
    time.sleep(random.uniform(0.01, 0.05))

save_contents(out_txt_file, tr_results)

print("Translation completed!")
