from flask import Flask, render_template
import os

app = Flask(__name__)

IMAGE_DIR = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR26_team_r_r4"


@app.route('/')
def index():
    images = []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            images.append(filename)

    return render_template('images.html', images=images)


if __name__ == '__main__':
    app.run(debug=True)
