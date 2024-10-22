import cv2

from ultralytics.models import YOLO
from ultralytics.models import RTDETR

cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)

# Load the YOLO11 model
# model = YOLO("yolo11n.pt")
model = RTDETR()
# Open the video file
video_path = "test_videos/xueshi_new_211.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, tracker='botsort_soccer.yaml', persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # cv2.waitkey(0)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
