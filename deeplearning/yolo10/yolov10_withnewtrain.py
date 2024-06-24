from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10(f'./weights/best.pt')

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=f"./tumor/valid/images",
    annotations_directory_path=f"./tumor/valid/labels",
    data_yaml_path=f"./tumor/data.yaml"
)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

import random

random_image = random.choice(list(dataset.images.keys()))
random_image = dataset.images[random_image]

results = model(source=random_image, conf=0.25)[0]
detections = sv.Detections.from_ultralytics(results)

annotated_image = bounding_box_annotator.annotate(
    scene=random_image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
