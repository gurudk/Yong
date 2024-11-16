from moviepy.editor import *

clip = VideoFileClip("/home/wolf/datasets/screenrecorder/src/SR10301.mp4")
newclip = clip.rotate(90)
newclip.write_videofile("/home/wolf/datasets/screenrecorder/src/SR11302.mp4")
