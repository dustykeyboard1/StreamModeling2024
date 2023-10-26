import sys
import os

from PySide6.QtCore import Qt, QDir
from PySide6.QtWidgets import QFileDialog, QSlider, QCheckBox, QApplication, QMainWindow, QPushButton, QGridLayout, QLabel, QWidget, QVBoxLayout, QLineEdit, QFormLayout, QHBoxLayout
from PySide6.QtGui import QPixmap

STARTER_ROWS = 16
GUI_WIDTH = 500
GUI_HEIGHT = 747
LINEEDIT_WIDTH = 330
FILE_INPUT_TEXT_WIDTH = 250
BROWSE_BUTTON_WIDTH = 75
ERROR_COLOR = "red"

class TitleBanner(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("QLabel {font-size : 16pt; font-weight: bold;}")
        self.setAlignment(Qt.AlignCenter)

class ErrorMessage(QLabel):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setStyleSheet("QLabel { color : " + ERROR_COLOR + " }")

class SettingsInput(QLineEdit):
    def __init__(self, label, placeholdertext):
        super().__init__()
        self.setPlaceholderText(placeholdertext)
        self._label = label
        self.setFixedWidth(LINEEDIT_WIDTH)
    
    def get_text(self):
        return self.text()

    def get_placeholdertext(self):
        return self.placeholderText()

    def get_label(self):
        return self._label

class FileInput(QLineEdit):
    def __init__(self, label, placeholder):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self._label = label
        self.setFixedWidth(FILE_INPUT_TEXT_WIDTH)
    
    def get_text(self):
        return self.text()

    def get_placeholdertext(self):
        return self.placeholderText()

    def get_label(self):
        return self._label

class SubmitButton(QPushButton):
    def __init__(self, form, input_filename, settings, sensitivities, output_settings, output_file_location):
        super().__init__("Submit")
        self._form = form
        self._input_filename = input_filename
        self._output_filename = output_file_location
        self._settings = settings
        self._sensitivities = sensitivities
        self._output_settings = output_settings
        self._results = []
        self.clicked.connect(self.get_input)
    
    def get_input(self):
        self.clear_errors()
        self._results += self.get_input_filename()
        self._results += self.get_settings()
        self._results += self.get_sensitivities()
        self._results += self.get_output_settings()
        self._results += [self._output_filename.get_text()]
        if (self.validate_submission()):
            window.call_hflux(self._results)
        self._results = []
        
    def clear_errors(self):
        rows = self._form.rowCount()
        while (rows > STARTER_ROWS):
            self._form.removeRow(STARTER_ROWS + 1)
            rows -= 1

    def get_input_filename(self):
        filename = self._input_filename.get_text()
        return [filename]
        
    def get_settings(self):
        results = []
        for l in self._settings:
            value = l.get_text()
            if value != '1' and value != '2' and value != "":
                if len(value) > 20:
                    self._form.addRow(ErrorMessage("Incorrect Value: " + value[:20] + "... in setting: " + l.get_label()))
                else:
                    self._form.addRow(ErrorMessage("Incorrect Value: " + value + " in setting: " + l.get_label()))
            results.append(l.get_text())
        return results
    
    def get_sensitivities(self):
        results = []
        for sens in self._sensitivities:
            results.append(sens.value())
        return results

    def get_output_settings(self):
        results = []
        for out in self._output_settings:
            results.append(out.isChecked())
        return results
    
    def validate_submission(self):
        filename = self._results[0]
        if not os.path.exists(filename):
            self._form.addRow(ErrorMessage("Input path/file must be valid"))
            return False
        
        settings = self._results[1:5]
        for val in settings:
            if val != '2' and val != '1' and val != "":
                return False
        
        output_path = self._results[-1]
        save_to_pdf = self._results[-2]
        if save_to_pdf and (not os.path.exists(output_path)):
            self._form.addRow(ErrorMessage("Save to PDF was selected, but an invalid path was provided"))
            return False
        return True

class SensitivitySlider(QSlider):
    def __init__(self, label, min, max, step, tick_pos, tick_interval):
        super().__init__(Qt.Horizontal)
        self._label = label
        self.setMinimum(min)
        self.setMaximum(max)
        self.setSingleStep(step)
        self.setTickPosition(tick_pos)
        self.setTickInterval(tick_interval)
        self.valueChanged.connect(self.update_value)
    
    def get_label(self):
        return self._label

    def update_value(self):
        self._label.setText(self._label.text()[:self._label.text().find(":")])
        self._label.setText(self._label.text() + ": " + str(self.value()))

class BrowseFileButton(QPushButton):
    def __init__(self, lineedit):
        super().__init__("Browse")
        self.clicked.connect(self.get_file)
        self.setFixedWidth(BROWSE_BUTTON_WIDTH)
        self._lineedit = lineedit

    def get_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Input File', QDir.rootPath() , '*.xlsx')
        self._lineedit.setText(file_name)

class BrowsePathButton(QPushButton):
    def __init__(self, lineedit):
        super().__init__("Browse")
        self.clicked.connect(self.get_file)
        self.setFixedWidth(BROWSE_BUTTON_WIDTH)
        self._lineedit = lineedit

    def get_file(self):
        path_name = QFileDialog.getExistingDirectory(self, "Select Output Location", QDir.rootPath())
        self._lineedit.setText(path_name)

# Subclass QMainWindow to customize your application's main window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HFLUX Stream Temperature Solver")
        
        self.setFixedWidth(GUI_WIDTH)
        self.setFixedHeight(GUI_HEIGHT)

        ### Creating the logo and title banner
        pixmap = QPixmap(os.path.join(os.getcwd(), "Demo", "hlfux_logo.png"))
        hflux_logo = QLabel()
        hflux_logo.setPixmap(pixmap)

        form = QFormLayout()
        form.addRow(TitleBanner("HFLUX Stream Temperature Solver"))
        form.addRow(hflux_logo)
        
        ### Creating the file input and browse button
        input_filename, browser = self.create_file_input("Required: Excel File", "Path")
        form.addRow(input_filename.get_label(), browser)

        ### Creating all input fields (settings, sliders, output options)
        method, equation1, equation2, equation3 = self.create_settings(form)
        sens1, sens2, sens3 = self.create_sensitivity_sliders(form)
        graphs, pdf = self.create_output_options(form)

        ### Creating the output path and browse button
        output_file_location, browser = self.create_path_output("File Path (if saving to PDF)", "Please enter a file path")
        form.addRow(output_file_location.get_label(), browser)

        ### Creating the submit button and adding it to the form
        submit = SubmitButton(form, input_filename, [method, equation1, equation2, equation3], [sens1, sens2, sens3], [graphs, pdf], output_file_location)
        form.addRow(submit)
        self.setLayout(form)

        ### Centering the GUI on the screen
        qr = self.frameGeometry()
        cp = self.screen().availableSize()
        self.move((cp.width() / 2) - (GUI_WIDTH / 2), (cp.height() / 2) - (GUI_HEIGHT / 2))
        
    def create_settings(self, form):
        method = SettingsInput("Solution Method", "Crank-Nicolson (1) or 2nd Order Runge-Kutta (2)")
        equation1 = SettingsInput("Shortwave Radiation Method", "Correction for Reflection (1) or Albedo Correction (2)")
        equation2 = SettingsInput("Latent Heat Flux Equation", "Penman (1) or Mass Transfer Method (2)")
        equation3 = SettingsInput("Sensible Heat Equation", "Bowen Ratio Method (1) or Dingman 1994 (2)")

        form.addRow(QLabel("\nEquation Settings. Providing no value defaults to the Excel Sheet"))
        form.addRow(method.get_label(), method)
        form.addRow(equation1.get_label(), equation1)
        form.addRow(equation2.get_label(), equation2)
        form.addRow(equation3.get_label(), equation3)

        return method, equation1, equation2, equation3
    
    def create_sensitivity_sliders(self, form):
        form.addRow(QLabel("\nSensitivity Sliders"))
        sens1 = SensitivitySlider(QLabel("Sensitivity 1"), min=0, max=10, step=1, tick_pos=QSlider.TicksBelow, tick_interval=1)
        form.addRow(sens1.get_label(), sens1)

        sens2 = SensitivitySlider(QLabel("Sensitivity 2"), min=0, max=15, step=1, tick_pos=QSlider.TicksBelow, tick_interval=1)
        form.addRow(sens2.get_label(), sens2)

        sens3 = QSlider(Qt.Horizontal)
        sens3 = SensitivitySlider(QLabel("Sensitivity 3"), min=0, max=100, step=2, tick_pos=QSlider.TicksBelow, tick_interval=1)
        form.addRow(sens3.get_label(), sens3)

        return sens1, sens2, sens3
    
    def create_output_options(self, form):
        form.addRow(QLabel("\nOutput Options"))

        graphs = QCheckBox("Display Graphs")
        pdf = QCheckBox("Save to PDF")

        form.addRow(graphs)
        form.addRow(pdf)

        return graphs, pdf

    def create_file_input(self, label, placeholdertext):
        filebrowser = QHBoxLayout()        
        input_filename = FileInput(label, placeholdertext)
        filebrowser.addWidget(input_filename)
        filebrowser.addWidget(BrowseFileButton(input_filename))
        return input_filename, filebrowser

    def create_path_output(self, label, placeholdertext):
        filebrowser = QHBoxLayout()        
        output_path = FileInput(label, placeholdertext)
        filebrowser.addWidget(output_path)
        filebrowser.addWidget(BrowsePathButton(output_path))
        return output_path, filebrowser

    def call_hflux(self, settings_input):
        # Make a full call to hflux
        print(settings_input)
        
app = QApplication(sys.argv)

window = MainWindow()
window.show()
sys.exit(app.exec())