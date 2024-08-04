import datetime
import sys
import os
import json
from functools import cmp_to_key
from pathlib import Path
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap, QMouseEvent, QPainter
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

ANNOTATED_FILE = "annotations.txt"


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

        self.imagefiles = []
        self.annotated_json = {}
        nowtime = datetime.datetime.now()
        self.annotated_local_file = "./annotated/annotated.json." + nowtime.strftime("%Y%m%d%H%M%S")
        Path(self.annotated_local_file).touch()

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
        self.log_label.setText(self.file_name)

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
        self.log_label.setText(self.file_name)

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

    def image_release_event(self, event):
        if isinstance(event, QMouseEvent):
            self.annotated_json[self.file_name] = str(self.x1) + "," + str(self.y1) + "," + str(self.x2) + "," + str(
                self.y2)
            self.is_drawing = False
            print(self.annotated_json)
            with open(self.annotated_local_file, 'w') as f:
                f.write(json.dumps(self.annotated_json))

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

        for root, dirs, files in os.walk(self.dir_name):
            for file_name in files:
                self.imagefiles.append(file_name)

        self.imagefiles = sorted(self.imagefiles, key=cmp_to_key(compare_function))

        self.annotation_file = self.dir_name + "/" + ANNOTATED_FILE
        anno_path = Path(self.annotation_file)
        if not anno_path.is_file():
            anno_path.touch()

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
        return super(MainWindow, self).closeEvent(event)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
