from ultralytics.data.annotator import auto_annotate

auto_annotate(data="/home/wolf/datasets/screenrecorder/SR583/frames/SR583_0", det_model="yolo11n.pt",
              sam_model="sam2_b.pt")
