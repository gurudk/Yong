import os

import cv2

from pathlib import Path

from ultralytics.models import YOLO
from ultralytics.models import RTDETR


def split_videos(input_file, output_dir, frames_per_video=1500):
    gen_video_index = 0
    output_dir_path = Path(output_dir)

    input_file_stem = Path(input_file).stem
    output_dir_path = output_dir_path.joinpath(input_file_stem)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_file = output_dir_path.joinpath(
        input_file_stem + "_" + str(gen_video_index) + ".mp4")

    cap = cv2.VideoCapture(input_file)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    frame_index = 0
    # Read and write frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        # print(frame_index)
        # Process the frame (optional)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_index += 1

        if frame_index % frames_per_video == 0:
            out.release()
            print("Split video ", str(output_file), " generated")

            gen_video_index += 1
            output_file = output_dir_path.joinpath(input_file_stem + "_" + str(gen_video_index) + ".mp4")
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    # Release the video capture and writer objects
    cap.release()
    out.release()


def batch_generator(model, video_path, out_dir, conf_threshold=0.7):
    video_parent_dir_stem = video_path.split("/")[-2]
    video_file_stem = Path(video_path).stem
    out_dataset_path = Path(out_dir).joinpath(video_parent_dir_stem)
    track_frame_idx = 0
    delta = 0

    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, tracker='./ultralytics_config/botsort_soccer.yaml', persist=True)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clses = results[0].boxes.cls.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                annotated_frame = results[0].plot()
                for box, track_id, conf, cls in zip(boxes, track_ids, confs, clses):
                    if conf < conf_threshold or cls != 0:
                        continue
                    x, y, w, h = box
                    x_left = int(x - w / 2 - delta)
                    y_top = int(y - h / 2 - delta)
                    w = int(w + delta)
                    h = int(h + delta)
                    # print(track_id, x, y, w, h, conf)

                    player_path = out_dataset_path.joinpath(video_file_stem).joinpath(
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


def get_only_files_in_current(input_dir):
    input_path = Path(input_dir)
    return list([str(f) for f in input_path.iterdir() if f.is_file()])


# Load the YOLO11 model
model = YOLO("./ultralytics_models/yolo11n.pt")

# split_video_output_dir = "/home/wolf/datasets/screenrecorder/dest/"
# src_video_dir = "/home/wolf/datasets/screenrecorder/src/"
# track_player_images_out_dir = "/home/wolf/datasets/reid/DFL/dest"
#
# print(get_only_files_in_current(src_video_dir))
#
# for video_path in get_only_files_in_current(src_video_dir):
#     print("============================Begin generate:", video_path, "=====================================")
#     split_videos(video_path, split_video_output_dir)
#     print(video_path, " is generated===========================================")

video_base_path = "/home/wolf/datasets/screenrecorder/dest"
# video_path = "/home/wolf/datasets/screenrecorder/dest/SR33/SR33_0.mp4"
# track_player_images_out_dir = "/home/wolf/datasets/reid/DFL/dest/"
autogen_images_out_dir = "/home/wolf/datasets/reid/DFL/dest_auto/"
# batch_generator(model, video_path, track_player_images_out_dir)

splits = (4,)
for subdir in os.listdir(video_base_path):
    if "SR694" not in subdir:
        continue
    for s in splits:
        video_path = video_base_path + "/" + subdir + "/" + subdir + "_" + str(s) + ".mp4"
        batch_generator(model, video_path, autogen_images_out_dir)
        print(video_path, "is completed~")
