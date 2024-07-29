from PIL import Image

img = Image.open("sofa1_foreground_1920x1280.png")
imgr = img.resize((960, 640))
imgr.save("sofa1_foreground_960x640.png")
