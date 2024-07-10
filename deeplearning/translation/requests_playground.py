import random
import time

import requests
import re
from pathlib import Path

paulgrahm_essays_url = "http://www.paulgraham.com/articles.html"

headers = {
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'referer': 'https://www.paulgraham.com/articles.html'}

response = requests.get(paulgrahm_essays_url, headers)

with open("paulgraham/articles.html", 'w') as f:
    f.write(response.text)

matches = re.findall(r"<a\s+href=\"([^\"]*)\">(\s*.*?\s*)</a>", response.text)

graham_dict = {}

for m in matches:
    if m[0].startswith("https"):
        graham_dict[m[0]] = re.sub("<[^>]*?>", "", m[1])
    else:
        graham_dict["https://paulgraham.com/" + m[0]] = re.sub("<[^>]*?>", "", m[1])

del graham_dict['https://paulgraham.com/index.html']
del graham_dict['https://paulgraham.com/rss.html']

# for a in graham_dict:
#     print(a)

print(len(graham_dict))
with open('paulgraham/essays.txt', 'w') as f:
    for a in graham_dict:
        f.write(a)
        f.write(",\"")
        f.write(graham_dict[a])
        f.write("\"\n")

for m in graham_dict:
    key = m
    value = graham_dict[m]
    dir_name = ""
    file_name = ""
    file_content = ""
    if value == "Lisp for Web-Based Applications" or value == "Chapter 1 of Ansi Common Lisp" or value == "Chapter 2 of Ansi Common Lisp":
        dir_name = value
        file_name = value + ".html"
    else:
        dir_name = re.split(r"/", key)[-1]
        file_name = dir_name

    response = requests.get(key, headers)
    file_content = response.text

    Path("paulgraham/").joinpath(dir_name).mkdir(parents=True, exist_ok=True)
    file_w = "paulgraham/" + dir_name + "/" + file_name
    with open(file_w, 'w') as f:
        f.write(file_content)
        print("Article \"" + value + "\" has been downloaded from " + key + ".")

    time.sleep(random.uniform(0.1, 0.7))
