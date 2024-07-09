import re

file_path = "paper/test_tr_output/On the Opportunities and Risks of Foundation Models_cn.txt"
infile = open(file_path, 'r')

lines = infile.readlines()
infile.close()

out_path = "paper/test_tr_output/On the Opportunities and Risks of Foundation Models_cn_clean.txt"
out_file = open(out_path, "w")
out_lines = []
for line in lines:
    str = re.sub(r"[\s]+", "", line)
    if str != "":
        line = re.sub(r"\[[^]]+\]", "", line)
        out_lines.append(line)
        out_file.write(line)

out_file.close()
print(out_lines)
