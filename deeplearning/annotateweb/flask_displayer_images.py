import paddle.fluid.libpaddle.eager.ops.legacy
from flask import Flask, send_from_directory, render_template, request, flash, url_for, session
import os
import json
import re
from pathlib import Path

app = Flask(__name__)
DIR_DICT = {}
IMAGE_DIR = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241122104231/"
player_json_tpl = {
    "head.hair": "pingtou",
    "skin": "white",
    "jersey_no": 9,
    "body.up.clothing": "tshirt",
    "body.up.clothing.color": "dusk blue",
    "body.down.pants": "shorts",
    "body.down.pants.color": "black",
    "body.down.stocking": 1,
    "body.down.stocking.color": "night blue",
    "shoes.color": "eggshell",

}


def get_annotated_keys(player_json):
    annotated_keys = []
    for ik in player_json.keys():
        annotated_keys.append(ik.split("/")[-1])
    return annotated_keys


def get_annotated_liststr(keylist):
    return ",".join(keylist)


def get_player_dict(dir):
    json_file = IMAGE_DIR + dir + "/" + dir + ".json"
    player_dict = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as rf:
            player_dict = json.loads(rf.read())

    return player_dict


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

    jsonstr = json.dumps(player_json_tpl)
    jsonstr = re.sub(r"([{},])", r"\1\n", jsonstr)
    return render_template('images_layout.html', images=images, dirs=dirs, currdir=currdir, jsonstr=jsonstr)


@app.route('/toc/<dir>')
def toc(dir):
    images = []
    for filename in os.listdir(IMAGE_DIR + dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            images.append(filename)

    dirs = []
    for sub_dir in os.listdir(IMAGE_DIR):
        dirs.append(sub_dir)

    jsonstr = json.dumps(player_json_tpl)
    jsonstr = re.sub(r"([{},])", r"\1\n", jsonstr)

    json_file = IMAGE_DIR + dir + "/" + dir + ".json"

    annotated_list = ''
    player_annotation_dict = get_player_dict(dir)
    if len(player_annotation_dict) > 0:
        annotated_list = get_annotated_liststr(get_annotated_keys(player_annotation_dict))

    return render_template('images_layout.html', images=images, dirs=dirs, currdir=dir, jsonstr=jsonstr,
                           annotated_list=annotated_list)


@app.route('/annotatedjson', methods=['POST'])
def commit_json():
    if request.method == 'POST':
        data = request.form
        print("posted data:", data)
        currdir = data["currdir"]
        json_str = data['json_area']
        selected_list = data['selected_list']
        player_annotation_dict = get_player_dict(currdir)
        if selected_list:
            file_list = data['selected_list'].split(",")
            for file in file_list:
                player_annotation_dict[IMAGE_DIR + currdir + "/" + file] = json_str

        obj = json.loads(json_str)
        # save annotation to file
        print(obj)
        if len(player_annotation_dict) > 0:
            with open(IMAGE_DIR + "/" + currdir + "/" + currdir + ".json", 'w') as wf:
                wf.write(json.dumps(player_annotation_dict))

        images = []
        for filename in os.listdir(IMAGE_DIR + currdir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                images.append(filename)

        dirs = []
        for sub_dir in os.listdir(IMAGE_DIR):
            dirs.append(sub_dir)

        annotated_list = get_annotated_liststr(get_annotated_keys(player_annotation_dict))
        return render_template('images_layout.html', images=images, dirs=dirs, currdir=currdir, jsonstr=json_str,
                               annotated_list=annotated_list)


if __name__ == '__main__':
    app.config['IMAGE_DIR'] = IMAGE_DIR  # Specify the directory
    app.run(debug=True)
