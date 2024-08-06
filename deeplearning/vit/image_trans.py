from PIL import Image

img = Image.open("6.png")
imgr = img.resize((256, 144))
imgr.save("6_256.png")
