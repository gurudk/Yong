from flask import Flask, send_from_directory, render_template
import os
from pathlib import Path

app = Flask(__name__)
DIR_DICT = {}
IMAGE_DIR = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241122104231/"


@app.route('/images/<dir>/<filename>')
def display_image(dir, filename):
    return send_from_directory(app.config['IMAGE_DIR'] + dir, filename)


# @app.route('/images/<dir>/<filename>')
# def toc_display_image(dir, filename):
#     return send_from_directory(app.config['IMAGE_DIR'] + dir, filename)


@app.route('/')
def index():
    images = []
    currdir = "SR26_team_r_r4"
    for filename in os.listdir(IMAGE_DIR + currdir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            images.append(filename)
    dirs = []
    for sub_dir in os.listdir(IMAGE_DIR):
        dirs.append(sub_dir)

    return render_template('images_layout.html', images=images, dirs=dirs, currdir=currdir)


@app.route('/toc/<dir>')
def toc(dir):
    images = []
    for filename in os.listdir(IMAGE_DIR + dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            images.append(filename)

    dirs = []
    for sub_dir in os.listdir(IMAGE_DIR):
        dirs.append(sub_dir)

    return render_template('images_layout.html', images=images, dirs=dirs, currdir=dir)


if __name__ == '__main__':
    app.config['IMAGE_DIR'] = IMAGE_DIR  # Specify the directory
    app.run(debug=True)
