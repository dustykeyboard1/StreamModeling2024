import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSlider, QCheckBox, QApplication, QMainWindow, QPushButton, QGridLayout, QLabel, QWidget, QVBoxLayout, QLineEdit, QFormLayout, QHBoxLayout


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(500)
        self.setFixedWidth(500)

        hlayout = QHBoxLayout()
        vlayout = QVBoxLayout()

        form = self.form_creation(hlayout)

        self.setWindowTitle("Stream Modelling App")

        hlayout.addLayout(form)
        vlayout.addLayout(hlayout)

        widget = QWidget()
        widget.setLayout(vlayout)
        self.setCentralWidget(widget)
        


    '''
    Creates inputs for xlsx files to be submitted to the application
    '''
    def form_creation(self, layout):
        e1 = QLineEdit()
        e2 = QLineEdit()
        e3 = QLineEdit()
        e4 = QLineEdit()
        e5 = QLineEdit()

        e2.setPlaceholderText("Crank-Nicolson (1) or 2nd Order Runge-Kutta (2)")
        e3.setPlaceholderText("Correction for Reflection (1) or Albedo Correction (2)")
        e4.setPlaceholderText("Penman (1) or Mass Transfer Method (2)")
        e5.setPlaceholderText("Bowen Ratio Method (1) or Dingman 1994 (2)")

        flo = QFormLayout()
        flo.addRow("File Name",e1)
        flo.addRow("Solution Method",e2)
        flo.addRow("Shortwave Radiation Method",e3)
        flo.addRow("Latent Heat Flux Equation",e4)
        flo.addRow("Sensible Heat Equation",e5)

        flo = self.sensitivity_sliders(flo)
        flo = self.output_settings(flo)

        flo.addRow(QLabel(" "))
        flo.addRow(QPushButton("Submit"))

        return flo
    

    '''
    Creates sensitivity sliders for the various parameters
    '''
    def sensitivity_sliders(self, flo):
        
        sens_label = QLabel("\nSensitivity Sliders")
        flo.addRow(sens_label)
        sens1 = QSlider(Qt.Horizontal)
        sens1.setMinimum(0)
        sens1.setMaximum(25)
        sens1.setSingleStep(1)
        sens1.setTickPosition(QSlider.TicksBelow)
        sens1.setTickInterval(1)
        flo.addRow("Sensitivity 1", sens1)

        sens2 = QSlider(Qt.Horizontal)
        sens2.setMinimum(0)
        sens2.setMaximum(50)
        sens2.setSingleStep(1)
        sens2.setTickPosition(QSlider.TicksBelow)
        sens2.setTickInterval(1)
        flo.addRow("Sensitivity 2", sens2)

        sens3 = QSlider(Qt.Horizontal)
        sens3.setMinimum(0)
        sens3.setMaximum(5)
        sens3.setSingleStep(1)
        sens3.setTickPosition(QSlider.TicksBelow)
        sens3.setTickInterval(1)
        flo.addRow("Sensitivity 3", sens3)

        return flo

    def output_settings(self, flo):
        flo.addRow(QLabel("\nOutput Options"))

        graphs = QCheckBox()
        graphs.setText("Display Graphs")

        pdf = QCheckBox()
        pdf.setText("Save to PDF")

        file_location = QLineEdit()
        file_location.setPlaceholderText("Please enter a file path or browse")

        flo.addRow(graphs, pdf)
        flo.addRow("File Path", file_location)

        return flo


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()