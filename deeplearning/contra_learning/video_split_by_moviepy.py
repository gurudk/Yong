import os
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip


def split_video(filename, segment_length, output_dir):
    clip = VideoFileClip(filename)
    duration = clip.duration

    start_time = 0
    end_time = segment_length
    i = 1

    # Extract the filename without extension
    basename = os.path.basename(filename).split('.')[0]

    # Extract directory path
    dir_path = os.path.dirname(filename)

    output_path = os.path.join(dir_path, output_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while start_time < duration:
        output = os.path.join(output_path, f"{basename}_part{i}.mp4")
        ffmpeg_extract_subclip(filename, start_time, min(end_time, duration), targetname=output)
        start_time = end_time
        end_time += segment_length
        i += 1
    print(f'Video split into {i - 1} parts.')


if __name__ == "__main__":
    input_file_name = "/home/wolf/datasets/xueshifootball/xueshi_new.mp4"
    output_dir = "/home/wolf/datasets/xueshifootball/clips"
    video_path = input_file_name  # first argument from command line
    segment_length = 30  # second argument from command line
    output_dir = output_dir  # third argument from command line
    split_video(video_path, segment_length, output_dir)
