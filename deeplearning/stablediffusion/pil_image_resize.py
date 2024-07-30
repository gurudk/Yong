from PIL import Image

init_image = Image.open("6M0A3501.jpg")
init_image.resize((960, 640))
init_image.save("6M0A3501_960x640.jpg")
