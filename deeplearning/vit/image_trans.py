from PIL import Image
import time

start = time.time()
img = Image.open("6.png")
imgr = img.resize((640, 360))
end = time.time()
imgr.save("6_640.png")

print("Trans duration(s):", str(end - start))
