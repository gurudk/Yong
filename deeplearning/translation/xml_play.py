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
    str_content.append(title.firstChild.data + "\\n")

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
            str_content.append(str_seg + "\\n")

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
        str_content.append(str_head + "\\n")
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

            str_content.append(str_seg + "\\n")
            print(str_seg)

    return str_content


file_name = "paper/test_tei/Batch Normalization.xml"
content = get_content(file_name)
print(content)
