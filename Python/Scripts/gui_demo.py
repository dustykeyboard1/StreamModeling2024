import sys
from sys import platform
import os
import numpy as np
import datetime
import time

import multiprocessing

from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QDir, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QSlider,
    QCheckBox,
    QApplication,
    QPushButton,
    QLabel,
    QWidget,
    QLineEdit,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QProgressBar,
)
from PySide6.QtGui import QPixmap

### Matplotlib
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.backends.backend_pdf import PdfPages

# Dynamically find and set the root directory.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

script_dir = os.path.dirname(os.path.abspath(__file__))
src_core_heat_flux_path = os.path.join(script_dir, '..', 'src', 'Core')
sys.path.append(src_core_heat_flux_path)

from Python.src.Core.heat_flux import HeatFlux
from Python.src.Heat_Flux.hflux_sens import HfluxSens
from Python.src.Utilities.data_table_class import DataTable
from Python.src.Plotting.hflux_errors_plotting import create_hflux_errors_plots
from Python.src.Heat_Flux.hflux_sens import HfluxSens

STARTER_ROWS = 19
GUI_WIDTH = 500
GUI_HEIGHT = 747
LINEEDIT_WIDTH = 310
FILE_INPUT_TEXT_WIDTH = 250
BROWSE_BUTTON_WIDTH = 60
ERROR_COLOR = "red"
MAX_ERROR_VALUE_LENGTH = 20


class TitleBanner(QLabel):
    def __init__(self, text):
        """
        Creates a stylized QLabel to serve as the title banner

        Args:
            text (str): text to be displayed

        Returns:
            None
        """
        super().__init__(text)
        self.setStyleSheet("QLabel {font-size : 16pt; font-weight: bold;}")
        self.setAlignment(Qt.AlignCenter)


class ErrorMessage(QLabel):
    def __init__(self, text):
        """
        Creates a stylized QLabel to serve as an error message

        Args:
            text (str): text to be displayed

        Returns:
            None
        """
        super().__init__()
        self.setText(text)
        self.setStyleSheet("QLabel { color : " + ERROR_COLOR + " }")


class SettingsInput(QLineEdit):
    def __init__(self, label, placeholdertext):
        """
        Creates a stylized QLabel to serve as the title banner

        Args:
            label (str): the label (text to the left) of the line edit
            placeholdertext (str): text that appears in the line edit when no other text is present

        Returns:
            None
        """
        super().__init__()
        self.setPlaceholderText(placeholdertext)
        self._label = label
        self.setFixedWidth(LINEEDIT_WIDTH)

    def get_text(self):
        """
        Returns the text contained in the settingsinput line edit

        Args:
            None

        Returns:
            The text in the line edit
        """
        return self.text()

    def get_placeholdertext(self):
        """
        Returns the placeholder text contained in the settingsinput line edit

        Args:
            None

        Returns:
            The placeholder text in the line edit
        """
        return self.placeholderText()

    def get_label(self):
        """
        Returns the label contained in the settingsinput line edit

        Args:
            None

        Returns:
            The label of the line edit
        """
        return self._label


class FileInput(QLineEdit):
    def __init__(self, label, placeholder):
        """
        Creates a file input line edit to accept files

        Args:
            label (str): the label (text to the left) of the file input
            placeholdertext (str): text that appears in the file input when no other text is present

        Returns:
            None
        """
        super().__init__()
        self.setPlaceholderText(placeholder)
        self._label = label
        self.setFixedWidth(FILE_INPUT_TEXT_WIDTH)

    def get_text(self):
        """
        Returns the text contained in the file input line edit

        Args:
            None

        Returns:
            The text in the line edit
        """
        return self.text()

    def get_placeholdertext(self):
        """
        Returns the placeholder text contained in the file input line edit

        Args:
            None

        Returns:
            The placeholder text in the line edit
        """
        return self.placeholderText()

    def get_label(self):
        """
        Returns the label contained in the file input line edit

        Args:
            None

        Returns:
            The label of the line edit
        """
        return self._label


class SubmitButton(QPushButton):
    def __init__(
        self,
        form,
        input_filename,
        settings,
        sensitivities,
        output_settings,
        output_file_location,
    ):
        """
        Creating a button so we can submit our inputs to hflux

        Args:
            form (QFormLayout): The form that is the main layout of the MainWindow
            input_filename (str): The string of text in the file input line edit
            settings (array): An array of values that correspond to the inputs for various solution settings
            sensitivities (array): An array of values from the sensitivity sliders
            output_settings (array): An array containing the values (T/F) of the PDF and Graphs checkboxes
            output_file_location (str): The path to where the output files should be written to

        Returns:
            None
        """
        super().__init__("Run Calculations")
        self._form = form
        self._input_filename = input_filename
        self._output_filename = output_file_location
        self._settings = settings
        self._sensitivities = sensitivities
        self._output_settings = output_settings
        self._results = []
        self.clicked.connect(self.get_input)

    def get_input(self):
        """
        Builds up a results array of all the inputs we passed in

        Args:
            None

        Returns:
            None
        """
        self.clear_errors()
        self._results += self.get_input_filename()
        self._results += self.get_settings()
        self._results += self.get_sensitivities()
        self._results += self.get_output_settings()
        self._results += [self._output_filename.get_text()]
        if self.validate_submission():
            window.call_hflux(self._results)
        self._results = []

    def clear_errors(self):
        """
        Gets rid of error messages when we call Submit, so as to avoid lingering error messages

        Args:
            None

        Returns:
            None
        """
        rows = self._form.rowCount()
        while rows > STARTER_ROWS:
            self._form.removeRow(STARTER_ROWS + 1)
            rows -= 1

    def get_input_filename(self):
        """
        Returns the input filename the user submitted as an array

        Args:
            None

        Returns:
            A singleton array containing the filename
        """
        filename = self._input_filename.get_text()
        return [filename]

    def get_settings(self):
        """
        Returns the values from settings as an array. Also checks for erroneous values in these inputs

        Args:
            None

        Returns:
            An array containing valid settings input
        """
        results = []
        for l in self._settings:
            value = l.get_text()
            if value != "1" and value != "2" and value != "":
                if len(value) > MAX_ERROR_VALUE_LENGTH:
                    self._form.addRow(
                        ErrorMessage(
                            "Incorrect Value: "
                            + value[:MAX_ERROR_VALUE_LENGTH]
                            + "... in setting: "
                            + l.get_label()
                        )
                    )
                else:
                    self._form.addRow(
                        ErrorMessage(
                            "Incorrect Value: "
                            + value
                            + " in setting: "
                            + l.get_label()
                        )
                    )
            results.append(l.get_text())
        return results

    def get_sensitivities(self):
        """
        Returns the values from the sensitivity sliders in an array

        Args:
            None

        Returns:
            An array containing sensitivity input
        """
        results = []
        for sens in self._sensitivities:
            results.append(sens.value())
        return results

    def get_output_settings(self):
        """
        Returns an array of output settings

        Args:
            None

        Returns:
            An array containing output settings
        """
        results = []
        for out in self._output_settings:
            results.append(out.isChecked())
        return results

    def validate_submission(self):
        """
        Ensures that all inputs are valid. Currently, we check:
            1) The input file exists
            2) All settings values are either 1 or 2
            3) If save to pdf is selected, then a valid output path was provided
        Args:
            None

        Returns:
            True if inputs are valid, and False otherwise
        """
        filename = self._results[0]
        if not os.path.exists(filename):
            self._form.addRow(ErrorMessage("Input path/file must be valid"))
            return False

        settings = self._results[1:5]
        for val in settings:
            if val != "2" and val != "1" and val != "":
                return False

        output_path = self._results[-1]
        save_to_pdf = self._results[-3]
        save_to_csv = self._results[-2]
        if (save_to_pdf or save_to_csv) and (not os.path.exists(output_path)):
            self._form.addRow(
                ErrorMessage(
                    "Save to PDF/CSV was selected, but an invalid path was provided"
                )
            )
            return False
        return True


class SensitivitySlider(QSlider):
    def __init__(self, label, min, max, step, tick_pos, tick_interval):
        """
        Creates a sensitivity slider

        Args:
            label (str): The string that appears to the left of the slider
            min (int): The minimum value the slider can take on
            max (int): The maximum value the slider can take on
            step (int): The step size of the slider
            tick_pos (int): The location of ticks on the slider (above, below, etc)
            tick_interval (int): The interval between tick marks

        Returns:
            None
        """
        super().__init__(Qt.Horizontal)
        self._label = label
        self.setMinimum(min)
        self.setMaximum(max)
        self.setSingleStep(step)
        self.setTickPosition(tick_pos)
        self.setTickInterval(tick_interval)
        self.valueChanged.connect(self.update_value)

    def get_label(self):
        """
        Returns the label of the tick slider

        Args:
            None

        Returns:
            The label associated with the tick slider
        """
        return self._label

    def update_value(self):
        """
        Updates the label to reflect the current value of the sensitivity slider

        Args:
            None

        Returns:
            None
        """
        self._label.setText(self._label.text()[: self._label.text().find(":")])
        self._label.setText(self._label.text() + ": " + str(self.value()))


class BrowseFileButton(QPushButton):
    def __init__(self, lineedit):
        """
        Creates a button to browse the file system with

        Args:
            lineedit (QLineEdit): The lineedit object that this browse button is associated with

        Returns:
            None
        """
        super().__init__("Browse")
        self.clicked.connect(self.get_file)
        self.setFixedWidth(BROWSE_BUTTON_WIDTH)
        self._lineedit = lineedit

    def get_file(self):
        """
        Sets the lineedit text to the full file name (path included) that was selected by the user

        Args:
            None

        Returns:
            None
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", QDir.rootPath(), "*.xlsx"
        )
        self._lineedit.setText(file_name)


class BrowsePathButton(QPushButton):
    def __init__(self, lineedit):
        """
        Creates a button to browse the file system with (only searches for paths)

        Args:
            lineedit (QLineEdit): The lineedit object that this browse button is associated with

        Returns:
            None
        """
        super().__init__("Browse")
        self.clicked.connect(self.get_file)
        self.setFixedWidth(BROWSE_BUTTON_WIDTH)
        self._lineedit = lineedit

    def get_file(self):
        """
        Sets the lineedit text to the full path that was selected by the user

        Args:
            None

        Returns:
            None
        """
        path_name = QFileDialog.getExistingDirectory(
            self, "Select Output Location", QDir.rootPath()
        )
        self._lineedit.setText(path_name)


class HfluxGraph(QtWidgets.QMainWindow):
    def __init__(self, figure, figure_name):
        """
        Creates a graph to be displayed

        Args:
            figure (Figure): The Matplotlib figure to be displayed by the GUI
            figure_name (str): The name of the window title

        Returns:
            None
        """
        super(HfluxGraph, self).__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        sc = FigureCanvasQTAgg(figure=figure)
        toolbar = NavigationToolbar(sc, self)

        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle(figure_name)
        self.show()


class HfluxCalculations(QObject):
    finished = Signal()
    additional_finished = Signal()
    variable_progress = Signal(list)
    progress = Signal(str)

    def __init__(self, settings_input):
        """
        Handles hflux calculations once we submit our input parameters

        Args:
            settings_input (array): The array that contains all settings information

        Returns:
            None
        """
        super().__init__()
        self._settings_input = settings_input
        self._sensitivity_results = {}

    def run(self):
        """
        Runs all of the hflux calculations, displays graphs, and saves data

        Args:
            None

        Returns:
            None
        """
        try:
            self.progress.emit("Beginning Calculations, 0")
            run_sens = self._settings_input[-5]
            display_graphs = self._settings_input[-4]
            savepdf = self._settings_input[-3]
            savecsv = self._settings_input[-2]
            output_path = self._settings_input[-1]
            filename = self._settings_input[0]
            self.progress.emit("Reading Data, 3")
            data_table = DataTable(filename)
        except:
            self.progress.emit("File Reading Error. Operation aborted, 100")
            time.sleep(2)
            self.finished.emit()

        self.progress.emit("Finished Data Reading, 5")
        data_table.output_suppression = 1

        self.change_settings(data_table, self._settings_input)
        self.change_sensitivities(data_table, self._settings_input)

        self.progress.emit("Beginning Heat Flux Calculations, 8")

        try:
            heat_flux = HeatFlux(data_table)
            (
                temp_mod,
                matrix_data,
                node_data,
                flux_data,
            ) = heat_flux.crank_nicolson_method()
        except:
            self.progress.emit("HFLUX calculation error. Operation aborted, 100")
            time.sleep(2)
            self.finished.emit()

        if run_sens:
            self.progress.emit("Finished Heat Flux Calculations, 20")
        else:
            self.progress.emit("Finished Heat Flux Calculations, 70")

        try:
            temp_dt = heat_flux.calculate_temp_dt(temp_mod)
            temp = data_table.temp.transpose()
            rel_err = heat_flux.calculate_percent_relative_error(temp, temp_dt)
            me = heat_flux.calculate_mean_residual_error(temp, temp_dt)
            mae = heat_flux.calculate_mean_absolute_residual_error(temp, temp_dt)
            mse = heat_flux.calculate_mean_square_error(temp, temp_dt)
            rmse = heat_flux.calculate_root_mean_square_error(temp, temp_dt)
            nrmse = heat_flux.calculate_normalized_root_mean_square_error(rmse, temp)
        except:
            self.progress.emit("Error during HFLUX Errors. Operation aborted, 100")
            time.sleep(2)
            self.finished.emit()

        if run_sens:
            self.progress.emit("Finished Error Calculations, 25")
        else:
            self.progress.emit("Finished Error Calculations, 95")

        dist_temp = data_table.dist_temp
        dist_mod = data_table.dist_mod
        time_temp = data_table.time_temp
        time_mod = data_table.time_mod

        if run_sens:
            hflux_sens = HfluxSens(root_dir)
            high_low_dict = hflux_sens.hflux_sens(
                data_table, [-0.01, 0.01], [-2, 2], [-0.1, 0.1], [-0.1, 0.1]
            )
            self.progress.emit("Starting Sensitivity Calculations, 30")
            sens = hflux_sens.create_new_results(
                temp_mod, high_low_dict, output_suppression=False, multithread=True
            )
            self.progress.emit("Finished Sensitivity Calculations, 95")
        else:
            hflux_sens, high_low_dict, sens = None, None, None

        self.progress.emit("Creating Graphs and Writing to Output!, 95")
        time.sleep(0.5)
        self.variable_progress.emit(
            [
                temp_mod,
                flux_data,
                temp,
                temp_dt,
                dist_temp,
                dist_mod,
                time_temp,
                time_mod,
                run_sens,
                data_table,
                sens,
                savecsv,
                savepdf,
                output_path,
                rel_err,
                display_graphs,
                heat_flux,
            ]
        )
        self.finished.emit()

    def change_settings(self, data_table, settings_input):
        """
        Changes the data_table from our excel file based on the user's input

        Args:
            data_table (DataTable): the data table that contains all information in the user's Excel file
            settings_input (array): An array of input from the user detailing which methods they want

        Returns:
            None
        """
        eq1 = settings_input[1]
        eq2 = settings_input[2]
        eq3 = settings_input[3]
        eq4 = settings_input[4]

        if eq1 != "":
            data_table.settings["solution method"] = int(eq1)

        if eq2 != "":
            data_table.settings["shortwave radiation method"] = int(eq2)

        if eq3 != "":
            data_table.settings["latent heat flux equation"] = int(eq3)

        if eq4 != "":
            data_table.settings["sensible heat equation"] = int(eq4)

    def change_sensitivities(self, data_table, settings_input):
        air_temp = int(settings_input[5])
        water_temp = int(settings_input[6])
        shade = int(settings_input[7])

        data_table.met_data["Air Temperature"] = data_table.met_data[
            "Air Temperature"
        ] * (1 + (air_temp / 100.0))
        data_table.t_l_data["Temperature"] = data_table.t_l_data["Temperature"] * (
            1 + (water_temp / 100.0)
        )
        data_table.shade_data["Shade "] = np.clip(
            data_table.shade_data["Shade "] * (1 + (shade / 100.0)), 0, 1
        )


class ProgressWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(90)
        self.setFixedWidth(300)
        self.setWindowTitle("Calculation Progress")
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setGeometry(200, 80, 250, 20)
        self._layout = QFormLayout()
        self._layout.addRow(self._progress_bar)
        self.setLayout(self._layout)
        self.show()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QWidget):
    def __init__(self):
        """
        Creates the Main Window of the GUI

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.setWindowTitle("HFLUX Stream Temperature Solver")

        self.setFixedWidth(GUI_WIDTH)
        self.setFixedHeight(GUI_HEIGHT)

        ### Creating the logo and title banner
        pixmap = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hflux_logo.png"))
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
        sens, graphs, pdf, csv = self.create_output_options(form)

        ### Creating the output path and browse button
        output_file_location, browser = self.create_path_output(
            "File Path (if saving to PDF)", "Please enter a file path"
        )
        form.addRow(output_file_location.get_label(), browser)

        ### Creating the submit button and adding it to the form
        submit = SubmitButton(
            form,
            input_filename,
            [method, equation1, equation2, equation3],
            [sens1, sens2, sens3],
            [sens, graphs, pdf, csv],
            output_file_location,
        )
        form.addRow(submit)
        self.setLayout(form)

        ### Centering the GUI on the screen
        qr = self.frameGeometry()
        cp = self.screen().availableSize()
        self.move(
            (cp.width() / 2) - (GUI_WIDTH / 2), (cp.height() / 2) - (GUI_HEIGHT / 2)
        )

    def create_settings(self, form):
        """
        Creates the settings input for the GUI

        Args:
            form (QFormLayout): The main form layout for the GUI

        Returns:
            method (SettingsInput): The settingsinput that refers to the method
            equation1 (SettingsInput): The settingsinput that refers to the shortwave radiation method
            equation2 (SettingsInput): The settingsinput that refers to the latent heat flux equation
            equation3 (SettingsInput): The settingsinput that refers to the sensible heat equation

        """
        method = SettingsInput(
            "Solution Method", "Crank-Nicolson (1) or 2nd Order Runge-Kutta (2)"
        )
        equation1 = SettingsInput(
            "Shortwave Radiation Method",
            "Correction for Reflection (1) or Albedo Correction (2)",
        )
        equation2 = SettingsInput(
            "Latent Heat Flux Equation", "Penman (1) or Mass Transfer Method (2)"
        )
        equation3 = SettingsInput(
            "Sensible Heat Equation", "Bowen Ratio Method (1) or Dingman 1994 (2)"
        )

        form.addRow(
            QLabel(
                "\nEquation Settings. Providing no value defaults to the Excel Sheet"
            )
        )

        form.addRow(method.get_label(), method)
        form.addRow(equation1.get_label(), equation1)
        form.addRow(equation2.get_label(), equation2)
        form.addRow(equation3.get_label(), equation3)

        return method, equation1, equation2, equation3

    def create_sensitivity_sliders(self, form):
        """
        Creates the sensitivity sliders for the GUI

        Args:
            form (QFormLayout): The main form layout for the GUI

        Returns:
            sens1 (SensitivitySlider): The first sensitivityslider
            sens2 (SensitivitySlider): The second sensitivityslider
            sens3 (SensitivitySlider): The third sensitivityslider
        """
        form.addRow(
            QLabel("\nInput Sliders: Values Represent Percent Changes to the Parameter")
        )
        sens1 = SensitivitySlider(
            QLabel("Air Temperature: 0"),
            min=-50,
            max=50,
            step=5,
            tick_pos=QSlider.TicksBelow,
            tick_interval=5,
        )
        form.addRow(sens1.get_label(), sens1)

        sens2 = SensitivitySlider(
            QLabel("Water Temperature: 0"),
            min=-50,
            max=50,
            step=5,
            tick_pos=QSlider.TicksBelow,
            tick_interval=5,
        )
        form.addRow(sens2.get_label(), sens2)

        sens3 = QSlider(Qt.Horizontal)
        sens3 = SensitivitySlider(
            QLabel("Shade: 0"),
            min=-50,
            max=50,
            step=5,
            tick_pos=QSlider.TicksBelow,
            tick_interval=5,
        )
        form.addRow(sens3.get_label(), sens3)

        return sens1, sens2, sens3

    def create_output_options(self, form):
        """
        Creates the output options for the GUI

        Args:
            form (QFormLayout): The main form layout for the GUI

        Returns:
            graphs (QCheckBox): The checkbox corresponding to whether the user wants the graphs to display
            pdf (QCheckBox): The checkbox corresponding to whether the user wants to save the graphs to a PDF
        """
        form.addRow(QLabel("\nOutput Options (Saving data causes a freeze. Please be patient :)"))

        sens = QCheckBox("Run Sensitivity Calculations")
        graphs = QCheckBox("Display Graphs")
        graphs.setChecked(True)
        pdf = QCheckBox("Save Graphs to PDF")
        csv = QCheckBox("Save CSV data")

        form.addRow(sens)
        form.addRow(graphs)
        form.addRow(pdf)
        form.addRow(csv)

        return sens, graphs, pdf, csv

    def create_file_input(self, label, placeholdertext):
        """
        Creates the file input selector for the GUI

        Args:
            label (str): The label for a FileInput line edit
            placeholdertext (str): The placeholdertext for a FileInput line edit

        Returns:
            input_filename (FileInput): The file input object that contains the user's desired path
            filebrowser (BrowseFileButton): The browse file button that allows the user to browse
        """
        filebrowser = QHBoxLayout()
        input_filename = FileInput(label, placeholdertext)
        filebrowser.addWidget(input_filename)
        filebrowser.addWidget(BrowseFileButton(input_filename))
        return input_filename, filebrowser

    def create_path_output(self, label, placeholdertext):
        """
        Creates the file output selector for the GUI

        Args:
            label (str): The label for a FileInput line edit
            placeholdertext (str): The placeholdertext for a FileInput line edit

        Returns:
            output_path (FileInput): The file input object that contains the user's desired path
            filebrowser (BrowsePathButton): The browse path button that allows the user to browse
        """
        filebrowser = QHBoxLayout()
        output_path = FileInput(label, placeholdertext)
        filebrowser.addWidget(output_path)
        filebrowser.addWidget(BrowsePathButton(output_path))
        return output_path, filebrowser

    def call_hflux(self, settings_input):
        """
        Makes a call to hfluxCalculations

        Args:
            settings_input (array): An array of the user's input options

        Returns:
            None
        """

        self.pwindow = ProgressWindow()
        self.hflux_thread = QThread()
        self.hf = HfluxCalculations(settings_input)
        self.hf.moveToThread(self.hflux_thread)

        self.hflux_thread.started.connect(self.hf.run)
        self.hf.finished.connect(self.hflux_thread.quit)
        self.hf.finished.connect(self.hf.deleteLater)
        self.hflux_thread.finished.connect(self.hflux_thread.deleteLater)

        self.hf.progress.connect(self.hflux_update)
        self.hf.variable_progress.connect(self.io)
        self.hflux_thread.start()

    def io(self, data):
        ### Data collection
        temp_mod = data[0]
        flux_data = data[1]
        temp = data[2]
        temp_dt = data[3]
        dist_temp = data[4]
        dist_mod = data[5]
        time_temp = data[6]
        time_mod = data[7]
        run_sens = data[8]
        data_table = data[9]
        sens = data[10]
        savecsv = data[11]
        savepdf = data[12]
        output_path = data[13]
        rel_err = data[14]
        display_graphs = data[15]
        heat_flux = data[16]
        (
            hflux_resiudal,
            hflux_3d,
            hflux_subplots,
            comparison_plot,
        ) = heat_flux.create_hlux_plots(
            temp_mod=temp_mod,
            flux_data=flux_data,
            sub_directory_path="",
            return_graphs=True,
        )
        errorsfig1, errorsfig2 = create_hflux_errors_plots(
            (temp - temp_dt),
            dist_temp,
            temp,
            temp_mod,
            dist_mod,
            time_temp,
            time_mod,
            return_graphs=True,
        )

        if run_sens:
            hflux_sens = HfluxSens(root_dir)
            sensfig1, sensfig2 = hflux_sens.make_sens_plots(
                data_table, sens, return_graphs=True
            )
        else:
            sensfig1, sensfig2 = None, None

        self.delete_window()

        if savecsv or savepdf:
            self.savedata(
                output_path,
                run_sens,
                savepdf,
                savecsv,
                temp_mod,
                data_table,
                flux_data,
                rel_err,
                [hflux_resiudal, hflux_3d, hflux_subplots, comparison_plot],
                [errorsfig1, errorsfig2],
                [sensfig1, sensfig2],
            )

        if display_graphs:
            self.display_figures(
                [
                    run_sens,
                    hflux_resiudal,
                    hflux_3d,
                    hflux_subplots,
                    comparison_plot,
                    errorsfig1,
                    errorsfig2,
                    sensfig1,
                    sensfig2,
                ]
            )

    def savedata(
        self,
        path,
        run_sens,
        savepdf,
        savecsv,
        temp_mod,
        data_table,
        flux_data,
        rel_err,
        hflux_plots,
        errors_plots,
        sensitivity_plots,
    ):
        dt = datetime.datetime.now().strftime("%Y-%m-%d--%H%M%S")
        folder = "HF_" + dt
        path = os.path.join(path, folder)
        os.mkdir(path)
        if savecsv:
            self.savecsvs(path, temp_mod, data_table, flux_data, rel_err)
        if savepdf:
            self.savepdfs(path, run_sens, hflux_plots, errors_plots, sensitivity_plots)

    def savecsvs(self, path, temp_mod, data_table, flux_data, rel_err):
        np.savetxt(f"{path}/temp_mod.csv", temp_mod, delimiter=",")
        np.savetxt(f"{path}/temp.csv", data_table.temp, delimiter=",")
        np.savetxt(f"{path}/rel_err.csv", rel_err, delimiter=",")
        np.savetxt(f"{path}/heatflux_data.csv", flux_data["heatflux"], delimiter=",")
        np.savetxt(f"{path}/solarflux_data.csv", flux_data["solarflux"], delimiter=",")
        np.savetxt(
            f"{path}/solar_refl_data.csv", flux_data["solar_refl"], delimiter=","
        )
        np.savetxt(f"{path}/long_data.csv", flux_data["long"], delimiter=",")
        np.savetxt(f"{path}/atmflux_data.csv", flux_data["atmflux"], delimiter=",")
        np.savetxt(f"{path}/landflux_data.csv", flux_data["landflux"], delimiter=",")
        np.savetxt(f"{path}/backrad_data.csv", flux_data["backrad"], delimiter=",")
        np.savetxt(f"{path}/evap_data.csv", flux_data["evap"], delimiter=",")
        np.savetxt(f"{path}/sensible_data.csv", flux_data["sensible"], delimiter=",")
        np.savetxt(
            f"{path}/conduction_data.csv", flux_data["conduction"], delimiter=","
        )

    def savepdfs(self, pdfpath, run_sens, hflux_plots, errors_plots, sensitivity_plots):
        """
        Saves hflux graphs to pdfs

        Args:
            hflux_plots (array): An array of hflux plots
            errors_plots (array): An array of hflux_errors plots
            sensitivity_plots (array): An array of sensitivity plots

        Returns:
            None
        """
        hflux_pdf = PdfPages(os.path.join(pdfpath, "hflux.pdf"))
        errors_pdf = PdfPages(os.path.join(pdfpath, "hflux_errors.pdf"))

        for fig in hflux_plots:
            hflux_pdf.savefig(fig)

        for fig in errors_plots:
            errors_pdf.savefig(fig)

        hflux_pdf.close()
        errors_pdf.close()

        if run_sens:
            sensitivity_pdf = PdfPages(os.path.join(pdfpath, "hflux_sens.pdf"))
            for fig in sensitivity_plots:
                sensitivity_pdf.savefig(fig)
            sensitivity_pdf.close()

    def display_figures(self, figs):
        self.hflux_residual = HfluxGraph(figs[1], "2D Modelled Stream Temperature")
        self.hflux_3d = HfluxGraph(figs[2], "3D Modelled Stream Temperature")
        self.hflux_subplots = HfluxGraph(figs[3], "Modelled Heat Flux and Radiation")
        self.comparison_plot = HfluxGraph(figs[4], "Energy Models")
        self.errorsfig1 = HfluxGraph(figs[5], "Modelled Temperature Residuals")
        self.errorsfig2 = HfluxGraph(figs[6], "Stream Temperature")
        if figs[0]:
            self.sensfig1 = HfluxGraph(figs[7], "Sensitivity of Recorded Values")
            self.sensfig2 = HfluxGraph(figs[8], "Sensitivity of Inputs")

    def hflux_update(self, value):
        value_string, progress_val = value[: value.find(",")], int(
            value[value.find(",") + 2 :]
        )
        self.pwindow._layout.removeRow(1)
        self.pwindow._layout.addRow(QLabel(value_string))
        self.pwindow._progress_bar.setValue(progress_val)

    def delete_window(self):
        self.hflux_update("Finished!, 100")
        self.pwindow.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    if sys.platform.lower() == "darwin":
        app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
