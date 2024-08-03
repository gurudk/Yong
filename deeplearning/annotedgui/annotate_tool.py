import sys
import os
from pathlib import Path
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QFileDialog,
    QVBoxLayout,
    QWidget
)


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

    def onMyToolBarLastImageClick(self, s):
        print("last image!")
        self.statusBar().showMessage("I'm last!")

    def onMyToolBarNextImageClick(self, s):
        print("next image!")
        self.statusBar().showMessage("I'm next!")

    def image_press_event(self, event):
        print("mouse pressed in image")
        print(event)

    def image_move_event(self, event):
        if isinstance(event, QMouseEvent):
            print(event.x(), event.y())

    def open_file_menu_clicked(self, s):
        print("open file clicked")
        self.file_name = QFileDialog.getOpenFileName(self, self.tr("Open File"), "/home",
                                                     self.tr("Images (*.png *.xpm *.jpg)"))[0]
        self.dir_name = Path(self.file_name).parent
        print(self.dir_name)
        print(self.file_name)

        pixmap = QPixmap(self.file_name)
        print(pixmap.height(), pixmap.width())
        self.pixmap = pixmap.scaled(1280, int(1280 * (pixmap.height() / pixmap.width())))
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.mousePressEvent = self.image_press_event
        self.image_label.mouseMoveEvent = self.image_move_event
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.setWindowTitle(self.file_name)

        for root, dirs, files in os.walk(self.dir_name):
            for file_name in files:
                self.imagefiles.append(file_name)

        self.imagefiles = sorted(self.imagefiles, key=lambda s: (s, len(s)))

        for filename in self.imagefiles:
            print(filename)

    def mouseMoveEvent(self, event):
        # print('Mouse coords: ( %d : %d )' % (event.x(), event.y()))
        return super(MainWindow, self).mouseMoveEvent(event)

    def enterEvent(self, event):
        # print("Mouse entered~")
        return super(MainWindow, self).enterEvent(event)

    def leaveEvent(self, event):
        # print("Mouse left~")
        return super(MainWindow, self).leaveEvent(event)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
