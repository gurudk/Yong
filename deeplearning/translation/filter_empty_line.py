import re

file_path = "paper/test_tr_output/How To Do Great Work.txt"
infile = open(file_path, 'r')

lines = infile.readlines()
infile.close()

out_path = "paper/test_tr_output/How To Do Great Work_en_clean.txt"
out_file = open(out_path, "w")
out_lines = []
for line in lines:
    str = re.sub(r"[\s]+", "", line)
    if str != "":
        out_lines.append(line)
        out_file.write(line)

out_file.close()
print(out_lines)
