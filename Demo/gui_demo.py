import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QLabel, QWidget


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()

        self.setWindowTitle("Stream Modelling App")

        layout.addWidget(QLabel("Hi Professor Baker!"), 0, 0)
        layout.addWidget(QLabel("Submit Files"), 1, 1)
        layout.addWidget(QLabel("Adjust Sensitivity"), 1, 2)
        layout.addWidget(QPushButton("See Graphs"), 2, 1)
        layout.addWidget(QPushButton("Convert to PDF"), 2, 2)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()