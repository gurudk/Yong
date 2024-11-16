import cv2

# that's my original video - the one that I want to rotate 180 degrees
cap = cv2.VideoCapture('/home/wolf/datasets/screenrecorder/src/SR10301.mp4')

frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Get width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# here I'm trying to write the new rotated video
# Open the output video file before the loop, cv2.VideoWriter_fourcc(*"mp4v") = 0x7634706d
newvideoR = cv2.VideoWriter('/home/wolf/datasets/screenrecorder/src/SR11141.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 25,
                            (frame_height, frame_width))

print("帧数：", frame_number, "FPS:", fps)
# Original Frames
# frames = []
for i in range(frame_number):
    ret, frame = cap.read()
    # frames.append(frame)  # No need to append the original frames

    # here's where I try to rotate the video
    new = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # cv2.imshow('output', new)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    newvideoR.write(new)
    if i % 500 == 0:
        print(i, " frames processed~")

newvideoR.release()
cap.release()
print("Video processed completedly~")
