import random
import os

from pathlib import Path
from ai_extract_contents import save_contents

import sys
import uuid
import requests
import hashlib
from imp import reload

import time
import json

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = os.environ["netease_api_key"]
APP_SECRET = os.environ["netease_api_secret"]


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def translate():
    file_path = "paper/test_tr_output/On the Opportunities and Risks of Foundation Models.txt"
    base_name = Path(file_path).stem
    out_txt_file = "paper/test_tr_output/" + base_name + "_cn.txt"

    contents = get_source_text(file_path)

    tr_results = []
    for i in range(0, len(contents)):
        q = contents[i]

        data = {}
        data['from'] = 'en'
        data['to'] = 'zh-CHS'
        data['signType'] = 'v3'
        curtime = str(int(time.time()))
        data['curtime'] = curtime
        salt = str(uuid.uuid1())
        signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
        sign = encrypt(signStr)
        data['appKey'] = APP_KEY
        data['q'] = q
        data['salt'] = salt
        data['sign'] = sign
        data['vocabId'] = "您的用户词表ID"

        response = do_request(data)

        str_res = json.loads(str(response.content, 'utf-8'))
        # print(str_res)
        tr_result = str_res["translation"]
        tr_results.append(tr_result[0])
        print(tr_result[0])

        time.sleep(random.uniform(0.01, 0.02))

    save_contents(out_txt_file, tr_results)

    print("Translation completed!")


def get_source_text(file_path):
    infile = open(file_path, 'r')
    src_lines = []
    line = infile.readline()
    while line != "":
        src_lines.append(line)
        line = infile.readline()

    infile.close()
    return src_lines


if __name__ == '__main__':
    translate()
