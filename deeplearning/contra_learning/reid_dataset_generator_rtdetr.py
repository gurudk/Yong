import cv2

from pathlib import Path

from ultralytics.models import YOLO
from ultralytics.models import RTDETR

# cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)

# Load the YOLO11 model
model = YOLO("yolo11n.pt")
# model = RTDETR("rtdetr-l.pt")
# Open the video file
video_path = "/home/wolf/datasets/xueshifootball/no_sound_clips/xueshi_new_213.mp4"

# video_path = "/home/wolf/datasets/DFL/train/D35bd9041_1/D35bd9041_1 (44).mp4"
video_file_stem = Path(video_path).stem
reid_dataset_dir = "/home/wolf/datasets/reid/DFL/"
reid_dataset_path = Path(reid_dataset_dir)
track_frame_idx = 0
delta = 0

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, tracker='botsort_soccer.yaml', persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            annotated_frame = results[0].plot()
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x, y, w, h = box
                x_left = int(x - w / 2 - delta)
                y_top = int(y - h / 2 - delta)
                w = int(w + delta)
                h = int(h + delta)
                # print(track_id, x, y, w, h, conf)

                player_path = reid_dataset_path.joinpath(video_file_stem).joinpath(
                    video_file_stem + "_" + str(track_id))
                player_path.mkdir(parents=True, exist_ok=True)
                player_track_frame_file = player_path.joinpath(
                    video_file_stem + "_" + str(track_id) + "_frame_" + str(track_frame_idx) + "_" + str(
                        int(conf * 1000)) + ".png")

                box_crop = frame[y_top:y_top + h, x_left:x_left + w]
                cv2.imwrite(str(player_track_frame_file), box_crop)

            print("frame ", track_frame_idx, " is generated~")
        track_frame_idx += 1
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
