import sys
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
    QFileDialog
)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = "My Annotaed Tool"
        self.setWindowTitle(self.title)

        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        button_action = QAction(QIcon("arrow-180-medium.png"), "&Last", self)
        button_action.setStatusTip("Last Image")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        # button_action.setCheckable(True)
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QAction(QIcon("arrow-000-medium.png"), "&Next", self)
        button_action2.setStatusTip("Next Image")
        button_action2.triggered.connect(self.onMyToolBarButtonClick)
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

        self.label = QLabel(self)

    def onMyToolBarButtonClick(self, s):
        print("click", s)
        self.statusBar().showMessage("I'm ready!")

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
        self.pixmap = pixmap.scaled(1280, 720)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.label.mousePressEvent = self.image_press_event
        self.label.mouseMoveEvent = self.image_move_event
        self.setCentralWidget(self.label)
        # self.resize(pixmap.width(), pixmap.height())

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
