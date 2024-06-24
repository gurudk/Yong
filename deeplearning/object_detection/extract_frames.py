# Importing all necessary libraries
import cv2
import os

football_v = "/home/wolf/Downloads/xueshi.mp4"
xueshi_v = "xueshi/xueshi_video_1.mp4"
sample_mp4 = "talking_dog.mp4"
# Read the video from specified path
cam = cv2.VideoCapture(xueshi_v)
frame_skip = 100

try:

    # creating a folder named data
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):

    if currentframe > 300:
        break

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './frames/sample_frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
