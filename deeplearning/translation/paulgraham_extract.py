import re
from pathlib import Path

file_path = "paulgraham/better.html/better.html"


def process_html(file_path):
    basename = Path(file_path).stem
    out_file_path = "paulgraham/essays/" + basename + ".txt"

    file_contents = ""

    with open(file_path, 'r') as f:
        file_contents = f.read()

    title_match = re.findall(r"<title>(.*?)</title>", file_contents)
    print(title_match)

    clean_text = re.sub(r"<title>(.*?)</title>", "", file_contents)
    clean_text = re.sub(r"<script[^>]*?>(.|\s)*?</script>", "", clean_text)
    clean_text = re.sub(r"<b>Notes</b>", "<b>Notes:</b>", clean_text)
    clean_text = re.sub(r"<!--(.|\s)*?-->", "", clean_text)
    # clean_text = re.sub(r"<table[^>]*?>(.|\s)*?</table>", "", clean_text)
    clean_text = re.sub(r"\[[^]]*?\]", "", clean_text)
    clean_text = re.sub(r"(<br\s+/>)+", "\n", clean_text)
    clean_text = re.sub(r"<[^\s][^>]*>", "", clean_text)
    # clean_text = re.sub(r"\s+", " ", clean_text)
    clean_text = re.sub(r"[\r\n][\r\n]+", "\n", clean_text)
    clean_text = re.sub(r"\n+", " ", clean_text)
    clean_text = re.sub(r"[\.]", ".\n", clean_text)
    clean_text = re.sub(r"[\:]", ":\n", clean_text)
    clean_text = re.sub(r"[\?]", "?\n", clean_text)
    clean_text = re.sub(r"\n\s+", "\n", clean_text)

    # for email newline
    clean_text = re.sub(r"from\:", "\nfrom:", clean_text)
    clean_text = re.sub(r"to\:", "\nto:", clean_text)
    clean_text = re.sub(r"date\:", "\ndate:", clean_text)
    clean_text = re.sub(r"subject\:", "\nsubject:", clean_text)
    clean_text = re.sub(r"(\d+)\:\n(\d+)", r"\1:\2", clean_text)
    clean_text = re.sub(r"Re\:", "Reply:", clean_text)
    clean_text = re.sub(r"(\n\s[A-Z]\w+\s\d{4}\s+)", r"\1\n", clean_text)

    # recover double colon
    clean_text = re.sub(r"\.\n\"", ".\"\n", clean_text)

    # recover http addres
    clean_text = re.sub(r"(http\:)\n(//\w+\.)\n(\w+\s)", r"\1\2\3\n", clean_text)
    clean_text = re.sub(r"(http\:)\n(//\w+\.)\n(\w+\.)\n(\w+\s)", r"\1\2\3\4\n", clean_text)

    # recover E.g.
    clean_text = re.sub(r"\n(E\.)\n(g.)\n", r"\nE.g.", clean_text)

    # recover w.w.
    clean_text = re.sub(r"(\w+)\.\n(\w+)\.\n", r"\1.\2.\n", clean_text)

    # recover ...
    clean_text = re.sub(r"(\w+)(\.)\n(\.)\n(\.)(\")\n", r"\1\2\3\4\5", clean_text)

    # recover " W --> "\nW
    clean_text = re.sub(r"(\"\s)([A-Z])", r"\1\n\2", clean_text)

    with open(out_file_path, 'w') as fw:
        fw.write(title_match[0] + "\n")
        fw.write(clean_text)

    print(title_match[0] + "\n" + clean_text)


process_html(file_path)
