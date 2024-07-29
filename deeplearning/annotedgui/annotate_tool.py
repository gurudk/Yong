import sys
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)

        label = QLabel(self)
        pixmap = QPixmap('96.png')
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        self.setCentralWidget(label)
        self.resize(pixmap.width(), pixmap.height())

    def mouseMoveEvent(self, event):
        # print('Mouse coords: ( %d : %d )' % (event.x(), event.y()))
        return super(MainWindow, self).mouseMoveEvent(event)

    def enterEvent(self, event):
        print("Mouse entered~")
        return super(MainWindow, self).enterEvent(event)

    def leaveEvent(self, event):
        print("Mouse left~")
        return super(MainWindow, self).leaveEvent(event)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
