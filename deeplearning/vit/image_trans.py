from PIL import Image
import time

start = time.time()
img = Image.open("6.png")
imgr = img.resize((256, 144))
end = time.time()
imgr.save("6_256.png")

print("Trans duration(s):", str(end - start))
