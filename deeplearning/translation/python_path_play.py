from pathlib import Path

file_path = "paper/test_tei/Visutal Transformer.xml"

print(Path(file_path).parent.resolve().parent)
