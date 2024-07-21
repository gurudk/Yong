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


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
