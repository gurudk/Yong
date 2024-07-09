import os

from pathlib import Path

file_path = "paper/test_tei/tei_test_simple.xml"
out_txt_file = os.path.basename(file_path)
print(out_txt_file)

print(Path(file_path).stem)
