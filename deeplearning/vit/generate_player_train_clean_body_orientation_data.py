import json

import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from torchvision.ops import box_iou


# clean body orientation data
# 1. remove ball datasets
# 2. remove center player 有些是控球的player， 方向数据不正确。


def get_detection_results(image, model, processor, threshold=0.7):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        start_time = datetime.now().timestamp()
        outputs = model(**inputs)
        end_time = datetime.now().timestamp()
        # print("model execution time:", end_time - start_time)

    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]),
                                                      threshold=threshold)
    return results


def get_center_point(comma_str_coords):
    arr = comma_str_coords.split(",")
    int_arr = [int(s) for s in arr]

    return (int_arr[0] + int_arr[2]) // 2, (int_arr[1] + int_arr[3]) // 2


def get_annotated_box(comma_str_coords):
    arr = comma_str_coords.split(",")
    box = [int(s) for s in arr]

    return torch.tensor(box)


def get_target_point(target_box):
    xmin, ymin, xmax, ymax = target_box.tolist()

    return int(xmin + xmax) // 2, int(ymin)


LOCAL_MODEL_DIR = "./rtdetr_r50vd"
PLAYER_TRAIN_DIR = "/home/wolf/datasets/DFL/clean_body_orientation_atan2_07/"
PLAYTER_TRAIN_PATH = Path(PLAYER_TRAIN_DIR)
annotated_file = "./annotation/annotated.release.20240912113708.txt"
processor = RTDetrImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = RTDetrForObjectDetection.from_pretrained(LOCAL_MODEL_DIR)
player_detections = {}

nowtime = datetime.now()
player_annotated_file = "./player_annotation/clean_body_orientation_atan2_07.json." + nowtime.strftime("%Y%m%d%H%M%S")

with open(annotated_file, 'r') as f:
    json_dict = json.loads(f.read())

file_ids = {}
file_idx = 0
for key in json_dict.keys():
    # print(idx, key)
    file_ids[file_idx] = key
    file_idx += 1

delta = 5
file_idx = 0
for key in tqdm(json_dict.keys()):
    target_idx = 0
    focus_point = get_center_point(json_dict[key])
    img = Image.open(key)
    img = img.resize((1280, 720))
    results = get_detection_results(img, model, processor, threshold=0.7)
    # print(key)
    path_key = Path(key)
    img_stem_name = path_key.stem
    for result in results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            clazz = model.config.id2label[label.item()]
            target_point = get_target_point(box)

            if clazz in {"person"}:
                target_properties = {}
                (xmin, ymin, xmax, ymax) = box.tolist()

                target_tensor_box = torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0)
                # 如果标注和检测有重叠，则它是不适合作为身体朝向标注数据
                annotated_box = get_annotated_box(json_dict[key]).unsqueeze(0)
                if box_iou(target_tensor_box, annotated_box) > 0:
                    continue

                crop = img.crop((xmin - delta, ymin - delta, xmax + delta, ymax + delta))
                crop_dir = PLAYTER_TRAIN_PATH.joinpath(img_stem_name)
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir.joinpath(clazz + "_" + str(target_idx) + ".png")

                target_key = str(crop_path)
                target_properties["from_image"] = file_idx
                target_properties["focus_point"] = focus_point
                target_properties["target_point"] = target_point

                angle_x, angle_y = tuple(np.subtract(focus_point, target_point))

                angle_tan = np.arctan2(angle_y, angle_x)

                target_properties["target_angle"] = round(angle_tan, 4)

                player_detections[target_key] = target_properties

                crop.save(str(crop_path))

                target_idx += 1

                # print(
                #     f"Detected {model.config.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )
    file_idx += 1
    # break

all_dict = {}
all_dict["file_ids"] = file_ids
all_dict["player_detections"] = player_detections
with open(player_annotated_file, 'w') as wf:
    wf.write(json.dumps(all_dict))
