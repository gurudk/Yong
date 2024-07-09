from xml.dom import minidom
import re


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


def save_contents(file_path, contents):
    out_file = open(file_path, 'w')
    for line in contents:
        str = re.sub(r"[\s]+", "", line)
        if str != "":
            out_file.write(line + "\n")
    out_file.close()


file_path = "paper/test_tei/Token2Token ViT.xml"
en_text_file_path = "paper/test_tr_output/Token2Token ViT.txt"

contents = get_content(file_path)

save_contents(en_text_file_path, contents)

print(contents)
print(len(contents))
