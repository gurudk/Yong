import cv2
import numpy as np

# Create a blank image (black background)
# img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread("videos/frames/B1606b0e6_1 (34)/B1606b0e6_1 (34)_23.png")

# Define the polygon vertices
pts = np.array([[100, 100], [200, 50], [300, 100], [250, 200]], np.int32)
pts = pts.reshape((-1, 1, 2))

# Create a mask for the polygon
mask = np.zeros_like(img)
cv2.fillPoly(mask, [pts], (255, 255, 255))

# Create an overlay image with the desired color and transparency
overlay = np.zeros_like(img)
cv2.fillPoly(overlay, [pts], (255, 0, 255))  # Green color

# Apply the overlay to the original image using the mask
alpha = 0.25  # Transparency level (0.0 to 1.0)
result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Display the result
cv2.imshow("Transparent Polygon", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
