import re
from pathlib import Path
from fnmatch import fnmatch
import os
import shutil


def process_html(file_path):
    basename = Path(file_path).stem
    out_file_path = "paulgraham/essays/" + basename + ".txt"

    file_contents = ""

    with open(file_path, 'r') as f:
        file_contents = f.read()

    clean_text = re.sub(r"<title>(.*?)</title>", "", file_contents)
    clean_text = re.sub(r"<script[^>]*?>(.|\s)*?</script>", "", clean_text)
    clean_text = re.sub(r"(<br\s+/><br\s+/><b>)(.*)(</b><br\s+/><br\s+/>)", r"\1\2.\3", clean_text)
    clean_text = re.sub(r"<!--(.|\s)*?-->", "", clean_text)
    # clean_text = re.sub(r"<table[^>]*?>(.|\s)*?</table>", "", clean_text)
    clean_text = re.sub(r"\[[^]]*?\]", "", clean_text)
    clean_text = re.sub(r"<br\s*/>", "\n", clean_text)
    clean_text = re.sub(r"<[^\s][^>]*>", "", clean_text)
    clean_text = re.sub(r"[\r\n][\r\n]{2,}", "\n\n", clean_text)
    clean_text = re.sub(r"([^\.\n])\n([^\n])", r"\1 \2", clean_text)
    # clean_text = re.sub(r"\.\s{2,}([A-Z])", r".\n\1", clean_text)

    # deal with  &mdash;  —
    clean_text = re.sub("&mdash;", " - ", clean_text)

    # truncate with :
    clean_text = re.sub(r"([a-zA-Z]+\s*)\:\s*([a-zA-Z\"]+\s*)", r"\1:\n\2", clean_text)

    # truncate with ? and ?"
    clean_text = re.sub(r"([a-zA-Z]+\s*)\?[ \t]*([a-zA-Z]+\s*)", r"\1?\n\2", clean_text)
    clean_text = re.sub(r"([a-zA-Z]+\s*)\?\"\s*([a-zA-Z]+\s*)", r"\1?\"\n\2", clean_text)

    # shrink whitespace
    clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)

    # for email newline
    clean_text = re.sub(r"from\:", "\nfrom:", clean_text)
    clean_text = re.sub(r"to\:", "\nto:", clean_text)
    clean_text = re.sub(r"date\:", "\ndate:", clean_text)
    clean_text = re.sub(r"subject\:", "\nsubject:", clean_text)
    clean_text = re.sub(r"Re\:", "Reply:", clean_text)

    # deal with date
    clean_text = re.sub(r"(\s+[A-Z]\w+\s+\d{4})\s+([A-Z\(])", r"\1\n\n\2", clean_text)

    title_match = re.findall(r"<title>(.*?)</title>", file_contents)
    print(title_match)
    if title_match:
        with open(out_file_path, 'w') as fw:
            fw.write(title_match[0] + "\n")
            fw.write(clean_text)
        print(title_match[0] + "\n" + clean_text)
    else:
        with open(out_file_path, 'w') as fw:
            fw.write(clean_text)
        print(clean_text)


def get_html_files_by_dir(html_dir_path):
    list_file_paths = []
    for path, subdir, files in os.walk(dir_path):
        for name in files:
            if fnmatch(name, pattern):
                list_file_paths.append(os.path.join(path, name))

    return list_file_paths


def move_dirs(source_dir, target_dir):
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        if file_name.endswith(".html"):
            shutil.move(os.path.join(source_dir, file_name), target_dir)


dir_path = "paulgraham/sites"
pattern = "*.html"

for html_path in get_html_files_by_dir(dir_path):
    process_html(html_path)
    print(html_path + " get processed~")
