import os

from PIL import Image

import pytesseract

os.environ['TESSDATA_PREFIX'] = "/home/wolf/mygitcode/Yong/deeplearning/doc_ocr/tesseract/tessdata"
print(os.environ['TESSDATA_PREFIX'])
testdata_dir = "./testdata/"

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
print(pytesseract.image_to_string(Image.open(testdata_dir + 'test.png')))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
print(pytesseract.image_to_string(testdata_dir + 'test.png'))

# List of available languages
print(pytesseract.get_languages(config=''))

# French text image to string
print(pytesseract.image_to_string(Image.open(testdata_dir + 'test-european.jpg'), lang='fra'))

# Batch processing with a single file containing the list of multiple image file paths
print(pytesseract.image_to_string(testdata_dir + 'images.txt'))

# Timeout/terminate the tesseract job after a period of time
try:
    print(pytesseract.image_to_string(testdata_dir + 'test.jpg', timeout=2))  # Timeout after 2 seconds
    print(pytesseract.image_to_string(testdata_dir + 'test.jpg', timeout=0.5))  # Timeout after half a second
except RuntimeError as timeout_error:
    # Tesseract processing is terminated
    pass

# Get bounding box estimates
print(pytesseract.image_to_boxes(Image.open(testdata_dir + 'test.png')))

# Get verbose data including boxes, confidences, line and page numbers
print(pytesseract.image_to_data(Image.open(testdata_dir + 'test.png')))

# Get information about orientation and script detection
print(pytesseract.image_to_osd(Image.open(testdata_dir + 'test.png')))

# Get a searchable PDF
pdf = pytesseract.image_to_pdf_or_hocr(testdata_dir + 'test.png', extension='pdf')
with open('test.pdf', 'w+b') as f:
    f.write(pdf)  # pdf type is bytes by default

# Get HOCR output
hocr = pytesseract.image_to_pdf_or_hocr(testdata_dir + 'test.png', extension='hocr')
print(hocr)

# Get ALTO XML output
xml = pytesseract.image_to_alto_xml(testdata_dir + 'test.png')
print(xml)

print(pytesseract.image_to_string("./dqd/itemtypes/chuanzhong.png", lang='chi_sim'))
print(pytesseract.image_to_string("./dqd/itemtypes/kongqiulv.png", lang='chi_sim'))
print(pytesseract.image_to_string("./dqd/itemtypes/chuanqiuchenggonglv.png", lang='chi_sim'))

print(pytesseract.image_to_string("./dqd/items/kongqiulv.png", lang='chi_sim'))
