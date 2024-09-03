import datetime
import sys
import os
import json
import glob

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os import walk

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.ops.boxes import box_area
from torchvision.ops import box_iou
from torchvision.ops import generalized_box_iou, generalized_box_iou_loss

from functools import cmp_to_key
from pathlib import Path
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap, QMouseEvent, QPainter, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QFileDialog,
    QVBoxLayout,
    QWidget,

)
from PySide6 import QtGui

np.random.seed(0)
torch.manual_seed(0)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class SoccerDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.annotations = json.loads(f.read())
            self.dataarray = [(key, value) for key, value in self.annotations.items()]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataarray[idx][0]
        image = Image.open(img_name)
        # truth_tensor = torch.tensor(self.dataarray[idx][1]).reshape(-1, 4)
        truth_tensor = torch.tensor(self.dataarray[idx][1])
        sample = {'image': image, 'ground_truth': truth_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class SoccerViT(nn.Module):
    def __init__(self, chw, n_patches=40, n_blocks=6, hidden_d=256, n_heads=8, out_d=4):
        # Super constructor
        super(SoccerViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
                chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
                chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        patch_height = self.chw[1] // n_patches
        patch_width = self.chw[2] // n_patches
        patch_dim = self.chw[0] * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_d),
            nn.LayerNorm(hidden_d),
        )

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        # self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

        self.mlp = MLP(self.hidden_d, self.hidden_d, out_d, 3)

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        # patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        # tokens = self.linear_mapper(patches)
        tokens = self.to_patch_embedding(images)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]
        outputs_coord = self.mlp(out).sigmoid()
        return outputs_coord


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def get_predict_coordinate(file_path, soccer_model, transform, device):
    img = Image.open(file_path)
    x = transform(img)
    x = x[None, :]
    x = x.to(device)
    y_hat = soccer_model(x)

    y_hat_coord = box_cxcywh_to_xyxy(y_hat)

    return y_hat_coord


def get_truth_tensor(json_obj, file_path, device):
    ground_truth = json_obj[file_path]
    truth_tensor = torch.tensor(ground_truth)

    truth_tensor = truth_tensor[None, :]
    truth_tensor = truth_tensor.to(device)

    return truth_tensor


ANNOTATED_FILE = "annotations.txt"


def count_annotated():
    # Use the glob module to find all files in the current directory with a ".txt" extension.
    files = glob.glob("./annotated/*.*.*.*")

    # Sort the list of file names based on the modification time (getmtime) of each file.
    files.sort(key=os.path.getmtime)
    all_images = {}
    for file_name in files:
        with open(file_name, 'r') as fi:
            dic = json.loads(fi.read())
            for key in dic:
                all_images[key] = dic[key]

    return len(all_images)


def compare_function(s1, s2):
    if s1[0].isdigit() and not s2[0].isdigit():
        return -1
    elif not s1[0].isdigit() and s2[0].isdigit():
        return 1
    else:
        if len(s1) < len(s2):
            return -1
        elif len(s1) > len(s2):
            return 1
        else:
            if s1 < s2:
                return -1
            else:
                return 1


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = "My Annotaed Tool"
        self.setWindowTitle(self.title)
        self.resize(1280, 720)

        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        button_action = QAction(QIcon("arrow-180-medium.png"), "&Last", self)
        button_action.setStatusTip("Last Image")
        button_action.triggered.connect(self.onMyToolBarLastImageClick)
        # button_action.setCheckable(True)
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QAction(QIcon("arrow-000-medium.png"), "&Next", self)
        button_action2.setStatusTip("Next Image")
        button_action2.triggered.connect(self.onMyToolBarNextImageClick)
        button_action2.setShortcut(QKeySequence("n"))
        # button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        # toolbar.addWidget(QLabel("Hello"))
        # toolbar.addWidget(QCheckBox())

        self.setStatusBar(QStatusBar(self))

        file_open_action = QAction("&Open", self)
        file_open_action.triggered.connect(self.open_file_menu_clicked)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(file_open_action)
        # file_menu.addSeparator()

        # file_submenu = file_menu.addMenu("Submenu")
        # file_submenu.addAction(button_action2)
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.log_label = QLabel()
        widget = QWidget()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.log_label)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.log_label.setWordWrap(True)

        self.imagefiles = []
        self.annotated_json = {}
        nowtime = datetime.datetime.now()
        # self.annotated_local_file = "./annotated/annotated.json." + nowtime.strftime("%Y%m%d%H%M%S")
        # Path(self.annotated_local_file).touch()

        # Use the glob module to find all files in the current directory with a ".txt" extension.
        files = glob.glob("./annotated/*.*.*.*")

        # Sort the list of file names based on the modification time (getmtime) of each file.
        files.sort()

        # Load latest image
        with open(files[-1], 'r') as f:
            obj = json.loads(f.read())
            self.file_name = list(obj.keys())[-1]
            self.dir_name = Path(self.file_name).parent.absolute().as_posix()

            pixmap = QPixmap(self.file_name)
            self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
            self.image_label.setPixmap(self.pixmap)
            self.image_label.setScaledContents(True)
            self.image_label.mousePressEvent = self.image_press_event
            self.image_label.mouseMoveEvent = self.image_move_event
            self.image_label.mouseReleaseEvent = self.image_release_event
            self.resize(self.pixmap.width(), self.pixmap.height())
            self.setWindowTitle(self.file_name)

            for root, dirs, imagefiles in os.walk(self.dir_name):
                for file_name in imagefiles:
                    if file_name.lower().endswith("png") or file_name.lower().endswith("jpg"):
                        self.imagefiles.append(file_name)

            self.imagefiles = sorted(self.imagefiles, key=cmp_to_key(compare_function))

            self.annotation_file = self.dir_name + "/" + ANNOTATED_FILE
            anno_path = Path(self.annotation_file)
            if not anno_path.is_file():
                anno_path.touch()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('soccer_vit_1000.pth')

        self.transform = v2.Compose([
            # you can add other transformations in this list
            v2.ToImage(),
            v2.Resize((360, 640)),
            v2.ToDtype(torch.float32, scale=True)
        ])

        # load all annotated file into json_obj
        self.annotated_obj = {}
        self.normalized_dict = {}
        for file in files:
            with open(file, 'r') as infile:
                jsonobj = json.loads(infile.read())
                for key in jsonobj.keys():
                    self.annotated_obj[key] = jsonobj[key]

        for key, value in self.annotated_obj.items():
            arr = value.split(",")
            float_array = np.array([float(i) for i in arr])
            narr = np.round(float_array / np.array([1280, 720, 1280, 720]), 4)
            self.normalized_dict[key] = list(narr)
        print(self.normalized_dict)

    def onMyToolBarLastImageClick(self, s):
        print("last image!")
        self.statusBar().showMessage("I'm last!")

        curr_index = self.imagefiles.index(Path(self.file_name).name)
        next_filename = self.imagefiles[curr_index - 1]
        while next_filename.endswith(".txt"):
            curr_index = curr_index - 1
            next_filename = self.imagefiles[curr_index]

        print(curr_index, next_filename)
        self.file_name = self.dir_name + "/" + next_filename
        pixmap = QPixmap(self.file_name)
        self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.mousePressEvent = self.image_press_event
        self.image_label.mouseMoveEvent = self.image_move_event
        # self.image_label.repaint()
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.setWindowTitle(self.file_name)
        # self.log_label.setText(self.file_name)

        self.draw_predict_and_truth()

    def onMyToolBarNextImageClick(self, s):
        print("next image!")
        self.statusBar().showMessage("I'm next!")
        curr_index = self.imagefiles.index(Path(self.file_name).name)
        next_filename = self.imagefiles[curr_index + 1] if curr_index < (len(self.imagefiles) - 1) else self.imagefiles[
            0]

        while next_filename.endswith(".txt"):
            curr_index = curr_index + 1
            if curr_index < (len(self.imagefiles) - 1):
                next_filename = self.imagefiles[curr_index]
            else:
                next_filename = self.imagefiles[0]

        print(curr_index, next_filename)
        self.file_name = self.dir_name + "/" + next_filename
        pixmap = QPixmap(self.file_name)
        self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.mousePressEvent = self.image_press_event
        self.image_label.mouseMoveEvent = self.image_move_event
        # self.image_label.repaint()
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.setWindowTitle(self.file_name)
        # self.log_label.setText(self.file_name)

        self.draw_predict_and_truth()

    def image_press_event(self, event):
        if isinstance(event, QMouseEvent):
            self.x1 = event.x()
            self.y1 = event.y()
            self.is_drawing = True

    def image_move_event(self, event):
        if isinstance(event, QMouseEvent):
            pixmap = QPixmap(self.file_name)
            self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
            self.image_label.setPixmap(self.pixmap)

            self.x2 = event.x()
            self.y2 = event.y()

            pm = self.image_label.pixmap()
            painter = QPainter(pm)

            pen = QtGui.QPen()
            pen.setWidth(4)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)

            painter.drawRect(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
            painter.end()
            self.image_label.setPixmap(pm)

    def draw_predict_and_truth(self):

        y_hat_coord = get_predict_coordinate(self.file_name, self.model, self.transform, self.device)
        image_coord_tpl = torch.tensor([1280, 720, 1280, 720], device=self.device)
        image_coord_hat = (image_coord_tpl * y_hat_coord.squeeze()).to(dtype=int)

        pm = self.image_label.pixmap()
        painter = QPainter(pm)

        pen = QtGui.QPen()
        pen.setWidth(4)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)

        painter.drawRect(image_coord_hat[0], image_coord_hat[1], image_coord_hat[2] - image_coord_hat[0],
                         image_coord_hat[3] - image_coord_hat[1])

        # draw truth
        if self.file_name in self.normalized_dict:
            truth_tensor = get_truth_tensor(self.normalized_dict, self.file_name, self.device)
            image_coord_truth = (image_coord_tpl * truth_tensor.squeeze()).to(dtype=int)

            blue_pen = QtGui.QPen()
            blue_pen.setWidth(4)
            blue_pen.setColor(QtGui.QColor('blue'))
            painter.setPen(blue_pen)

            painter.drawRect(image_coord_truth[0], image_coord_truth[1], image_coord_truth[2] - image_coord_truth[0],
                             image_coord_truth[3] - image_coord_truth[1])

        painter.end()
        self.image_label.setPixmap(pm)

    def image_release_event(self, event):
        if isinstance(event, QMouseEvent):
            self.annotated_json[self.file_name] = str(self.x1) + "," + str(self.y1) + "," + str(self.x2) + "," + str(
                self.y2)
            self.is_drawing = False
            print(self.annotated_json)
            # with open(self.annotated_local_file, 'w') as f:
            #     f.write(json.dumps(self.annotated_json))
            log_str = str(list(self.annotated_json.items())[-1]) + "\n"
            log_str += str(count_annotated() + len(self.annotated_json)) + " images have been annotated," + str(
                len(self.imagefiles) - count_annotated() - len(
                    self.annotated_json)) + " images left~ \n"
            self.log_label.setText(log_str)

    def open_file_menu_clicked(self, s):
        print("open file clicked")
        self.file_name = QFileDialog.getOpenFileName(self, self.tr("Open File"), "/home",
                                                     self.tr("Images (*.png *.xpm *.jpg)"))[0]
        self.dir_name = Path(self.file_name).parent.absolute().as_posix()

        pixmap = QPixmap(self.file_name)
        self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.mousePressEvent = self.image_press_event
        self.image_label.mouseMoveEvent = self.image_move_event
        self.image_label.mouseReleaseEvent = self.image_release_event
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.setWindowTitle(self.file_name)

        if self.imagefiles is None:
            for root, dirs, files in os.walk(self.dir_name):
                for file_name in files:
                    if file_name.lower().endswith("png") or file_name.lower().endswith("jpg"):
                        self.imagefiles.append(file_name)

            self.imagefiles = sorted(self.imagefiles, key=cmp_to_key(compare_function))

        self.annotation_file = self.dir_name + "/" + ANNOTATED_FILE
        anno_path = Path(self.annotation_file)
        if not anno_path.is_file():
            anno_path.touch()

        self.draw_predict_and_truth()

    def mouseMoveEvent(self, event):
        # print('Mouse coords: ( %d : %d )' % (event.x(), event.y()))
        return super(MainWindow, self).mouseMoveEvent(event)

    def enterEvent(self, event):
        # print("Mouse entered~")
        return super(MainWindow, self).enterEvent(event)

    def leaveEvent(self, event):
        # print("Mouse left~")
        return super(MainWindow, self).leaveEvent(event)

    def mouseReleaseEvent(self, event):
        return super(MainWindow, self).mouseReleaseEvent(event)

    def closeEvent(self, event):
        print("App quit~")
        end_time = datetime.datetime.now()
        # session_file_name = self.annotated_local_file + "." + end_time.strftime("%Y%m%d%H%M%S")
        # with open(session_file_name, 'w') as sfile:
        #     sfile.write(json.dumps(self.annotated_json))

        return super(MainWindow, self).closeEvent(event)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
