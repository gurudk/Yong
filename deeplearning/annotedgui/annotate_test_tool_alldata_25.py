import sys
import glob

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

import json
import time
import datetime
import os
import smtplib
import traceback
from email.message import EmailMessage

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch import nn
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from einops.layers.torch import Rearrange

np.random.seed(0)
torch.manual_seed(0)


def send_mail(SUBJECT, TEXT):
    msg = EmailMessage()
    msg.set_content(TEXT)
    msg['Subject'] = SUBJECT
    msg['From'] = "soccervit@126.com"
    msg['To'] = "soccervit@126.com"

    s = smtplib.SMTP('smtp.126.com')
    s.login('soccervit@126.com', os.environ["auth_code"])
    s.send_message(msg)
    s.quit()


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


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


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class SoccerViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, out_dim=16, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(num_patches + 1, dim),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp = MLP(dim, dim, out_dim, 3)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.positional_embeddings.repeat(b, 1, 1)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp(x)


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
    block_index = int(torch.argmax(y_hat))

    row = block_index // 5
    col = block_index % 5

    x1 = col * 256.0
    y1 = row * 144.0

    return [x1, y1, x1 + 256, y1 + 144]


def get_predict_top3_blockindex(file_path, soccer_model, transform, device):
    img = Image.open(file_path)
    x = transform(img)
    x = x[None, :]
    x = x.to(device)
    y_hat = soccer_model(x)
    top3_index = torch.topk(y_hat, 3)[1].squeeze().tolist()

    return top3_index


def get_coord_from_block_index(block_index):
    row = block_index // 5
    col = block_index % 5

    x1 = col * 256.0
    y1 = row * 144.0

    return [x1, y1, x1 + 256, y1 + 144]


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

        for root, dirs, thesefiles in os.walk(self.dir_name):
            for file_name in thesefiles:
                if file_name.lower().endswith("png") or file_name.lower().endswith("jpg"):
                    self.imagefiles.append(file_name)

        self.imagefiles = sorted(self.imagefiles, key=cmp_to_key(compare_function))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('./model/std_vit_alldata_500_cross_1e4.pth')

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

        top3indexs = get_predict_top3_blockindex(self.file_name, self.model, self.transform, self.device)

        image_coord_tpl = torch.tensor([1280, 720, 1280, 720], device=self.device)

        pm = self.image_label.pixmap()
        painter = QPainter(pm)

        pen = QtGui.QPen()
        pen.setWidth(4)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)

        for block_index in top3indexs:
            temp_coord = get_coord_from_block_index(block_index)

            painter.drawRect(temp_coord[0], temp_coord[1], temp_coord[2] - temp_coord[0],
                             temp_coord[3] - temp_coord[1])

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

        # reset imagefiles in this folder ,align self.file_name, self.dir_name
        self.imagefiles = []
        for root, dirs, files in os.walk(self.dir_name):
            for file_name in files:
                if file_name.lower().endswith("png") or file_name.lower().endswith("jpg"):
                    self.imagefiles.append(file_name)

        self.imagefiles = sorted(self.imagefiles, key=cmp_to_key(compare_function))

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
