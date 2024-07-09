import json
import grobid_tei_xml

xml_path = "paper/test_tei/tei_test_simple.xml"

with open(xml_path, 'r') as xml_file:
    doc = grobid_tei_xml.parse_document_xml(xml_file.read())

# print(json.dumps(doc.to_dict(), indent=2))
print(doc.body)
