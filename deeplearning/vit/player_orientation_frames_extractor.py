# Importing all necessary libraries
import cv2
import os
import pathlib
from pathlib import Path


def generate_frames(video_file_path, out_dir_path):
    currentframe = 0
    cam = cv2.VideoCapture(str(video_file_path))
    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            video_file_stem = video_file_path.stem
            out_frames_dir_path = out_dir_path.joinpath(video_file_stem)
            if not out_frames_dir_path.exists():
                out_frames_dir_path.mkdir(parents=True, exist_ok=True)

            img_file_path = out_frames_dir_path.joinpath(video_file_stem + "_" + str(currentframe) + ".png")

            # writing the extracted images
            cv2.imwrite(str(img_file_path), frame)
            print(img_file_path, " created~")

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()


TEST_VIDEO_PATH = "./videos/20241002_2/D35bd9041_1 (40).mp4"
GEN_DIR = "./videos/frames"

# for root, dirs, files in os.walk(DFL_DIR):
#     print(sorted(files))
#     # print(dirs)

# root_dir = pathlib.Path(DFL_DIR)
#
# # for item in root_dir.iterdir():
# #     if item.is_file():
# #         print(item)
#
# allfiles = sorted(list(root_dir.rglob("*")))
#
# print(len(allfiles))
#
# for file_path in allfiles:
#     if file_path.suffix.endswith("mp4"):
#         vfile_dir = Path(GEN_DIR).joinpath(file_path.stem)
#         if not vfile_dir.exists():
#             vfile_dir.mkdir(parents=True, exist_ok=True)
#
#         # open the mp4
#         # print(str(file_path))
#         generate_frames(file_path, Path(GEN_DIR))
#         break

# def temp_gen_path(file_name):
#     stem = file_name.split(" ")[0]
#     train_dir = Path(DFL_DIR)
#     file_path = train_dir.joinpath(stem).joinpath(file_name)
#     return file_path


generate_frames(Path(TEST_VIDEO_PATH).resolve(), Path(GEN_DIR))
