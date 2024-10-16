import cv2
import glob

from pathlib import Path

SECONDS_PER_VIDEO = 1500


def split_videos(input_file, output_dir):
    gen_video_index = 0
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    input_file_stem = Path(input_file).stem
    output_file = output_dir_path.joinpath(input_file_stem + "_" + str(gen_video_index) + ".mp4")

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

        if frame_index % SECONDS_PER_VIDEO == 0:
            out.release()
            print("Split video ", str(output_file), " generated")

            gen_video_index += 1
            output_file = output_dir_path.joinpath(input_file_stem + "_" + str(gen_video_index) + ".mp4")
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    # Release the video capture and writer objects
    cap.release()
    out.release()


input_file_name = "/home/wolf/datasets/screenrecorder/SR583.mp4"
output_dir = "/home/wolf/datasets/screenrecorder/SR583/"

split_videos(input_file_name, output_dir)
