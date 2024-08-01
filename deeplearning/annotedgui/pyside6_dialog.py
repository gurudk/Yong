import sys

from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton
from PySide6.QtWidgets import QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press me for a dialog!")
        button.clicked.connect(self.button_clicked)
        self.setCentralWidget(button)

    def button_clicked(self, s):
        print("click", s)

        # dlg = QDialog(self)
        # dlg.setWindowTitle("HELLO!")
        # dlg.exec()

        # fileName = QFileDialog.getOpenFileName(self, self.tr("Open File"), "/home",
        #                                        self.tr("Images (*.png *.xpm *.jpg)"))

        fileName = QFileDialog.getSaveFileName(self, self.tr("Save F:xile"),
                                               "/home/jana/untitled.png",
                                               self.tr("Images (*.png *.xpm *.jpg)"))
        print(fileName)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
