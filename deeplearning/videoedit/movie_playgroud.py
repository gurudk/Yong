from moviepy.editor import *

clip = VideoFileClip("/home/wolf/datasets/xueshifootball/xueshi.mp4")
newclip = clip.rotate(90)
newclip.write_videofile("/home/wolf/datasets/xueshifootball/xueshi_new.mp4")
