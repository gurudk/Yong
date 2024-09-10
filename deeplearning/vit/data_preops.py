import json
from PIL import Image
import numpy as np

annotated_file = './annotation/annotated.release.20240910160828.txt'
normalized_annotation_file = "./annotation/annotation_normalized_20240910160828.txt"
normalized_dict = {}
with open(normalized_annotation_file, 'w') as wf:
    with open(annotated_file, 'r') as f:
        obj = json.loads(f.read())
        for key, value in obj.items():
            arr = value.split(",")
            float_array = np.array([float(i) for i in arr])
            narr = np.round(float_array / np.array([1280, 720, 1280, 720]), 4)
            normalized_dict[key] = list(narr)

    wf.write(json.dumps(normalized_dict))
