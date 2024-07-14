import easyocr

reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
result = reader.readtext('dqd/dqd_stat.jpg', text_threshold=0.5)
print(result)

for item in result:
    print(item[1])
