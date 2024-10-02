import glob

input_video_files_pattern = "./videos/*.mp4"
output_video_dir = "./videos/processed/"

input_video_path = glob.glob(input_video_files_pattern)

print(input_video_path)
