"""
Author: Michael Scoleri
File: Plotting_class.py
Date: 10-19-2023
Functionality: Construct a plotting class for plotting capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os


class Plotting:
    def __init__(self):
        """
        Initizlize the plotting class.
        """
        pass

    def class_assistant(self):
        """
        Print out class details.
        """
        help(type(self))

    def make3dplot(self, x, y, z, xlabel, ylabel, zlabel, colorbar_label, title):
        """
        Create and return a return 3d plot.

        Args: x (ndarray): The x-axis data.
                y (ndarray): The y-axis data.
                z (ndarray): The z-axis data.
                xlabel (string): The x-axis label.
                ylabel (string): The y-axis label.
                zlabel (string): The z-axis label.
                colorbar_label (string): The colorbar label.
                title (string): The title of the graph.

        Returns:
            Figure of 3d plot.
        """
        # 3D plot of stream temperature
        fig = plt.figure()

        # Create a 3D axis
        ax = fig.add_subplot(111, projection="3d")

        # Create a surface plot
        # Make x, y axis take different length - https://stackoverflow.com/questions/46607106/python-3d-plot-with-different-array-sizes
        x_sized, y_sized = np.meshgrid(x, y)
        # ax.plot_surface() - https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html#plot-surface-x-y-z

        surface = ax.plot_surface(
            x_sized, y_sized, z, cmap="jet", rstride=10, cstride=10
        )

        # Add a colorbar with label
        cbar = fig.colorbar(surface)
        cbar.set_label(colorbar_label, fontsize=11, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_zlabel(zlabel, fontsize=11)
        ax.invert_xaxis()

        return fig

    def make_single_plot(
        self,
        x,
        y,
        xlabel,
        ylabel,
        title,
        linewidth=None,
        marker=None,
        legend=None,
        axis=None,
        ax=None,
    ):
        """
        Creates a basic line plot.

        Args:
            x (ndarray): The x-axis data.
            y (ndarray): The y-axis data.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            title (str): The title of the graph.
            xlimit (int): The limit for the x-axis.
            ylimit (int): The limit for the y-axis.
            linewidth (float): The width of the line.
            marker (str): The marker style.
            legend ([str]): List of strings for the legend.
            axis ([ndarray]): axis parameters for plots.

        Returns:
            Figure: Figure of the residual plot.
        """
        ax.plot(x, y, marker)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.axis(axis)
        return ax

    def make_three_line_plot(self, x, low_y, base_y, high_y, title, xlabel, ylabel, ax):
        """
        Creates a 3 line plot for low, base and high.

        Args:
            low_x (ndarray): X data for low.
            low_y (ndarray): Y data for low.
            base_x (ndarray): X data for base.
            base_y (ndarray): Y data for base.
            high_x (ndarray): X data for high.
            high_y (ndarray): Y data for high.
            xlimit ([ndarray]): limit for the X axis.
            ylimit ([ndarray]): limit for the Y axis.

        Return:
            Figure with plotted data.
        """
        ax.plot(x, low_y, "--b", linewidth=2)
        ax.plot(x, base_y, "k", linewidth=2)
        ax.plot(x, high_y, "--r", linewidth=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(["Low", "Base", "High"])
        ax.set_xlim([np.min(x), np.max(x)])
        return ax

    def make_residual_plot(self, data, xlabel, ylabel, title, colorbar_label, extent):
        """
        Creates a residual plot (heat map).

        Args:
            data (ndarray): The heatmap data.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            title (str): The title of the graph.
            colorbar_label (str): The label for the colorbar.
            extent (list[float]): List of floats to set the extent of the graph.

        Returns:
            Figure: Figure of line plot.
        """

        fig, ax = plt.subplots()
        plt.imshow(data, aspect="auto", cmap="jet", origin="lower", extent=extent)
        plt.colorbar(label=colorbar_label)
        plt.title(title, fontsize=11, fontweight="bold")
        plt.xlabel(xlabel, fontsize=9)
        plt.ylabel(ylabel, fontsize=9)
        plt.tight_layout()
        ax.invert_yaxis()
        return fig

    def heat_flux_comparison(
        self,
        x,
        y1,
        marker1,
        label1,
        y2,
        marker2,
        label2,
        y3,
        marker3,
        label3,
        y4,
        marker4,
        label4,
        y5,
        marker5,
        label5,
        y6,
        marker6,
        label6,
        title,
        xlabel,
        ylabel,
    ):
        """
        Creates the heat flux camparison plot.

        Args:
            x (ndarry): x data for the plot.
            y1 (ndarry): the first set of y data.
            marker1 (char): type of marker to use for y1.
            label1 (str): the label for y1 data.
            y2 (ndarry): the second set of y data.
            marker2 (char): type of marker to use for y2.
            label2 (str): the label for y2 data.
            y3 (ndarry): the third set of y data.
            marker3 (char): type of marker to use for y3.
            label3 (str): the label for y3 data.
            y4 (ndarry): the fourth set of y data.
            marker4 (char): type of marker to use for y4.
            label4 (str): the label for y4 data.
            y5 (ndarry): the fifth set of y data.
            marker5 (char): type of marker to use for y5.
            label5 (str): the label for y5 data.
            y6 (ndarry): the sixth set of y data.
            marker6 (char): type of marker to use for y6.
            label6 (str): the label for y6 data.
            title (str): the title of the plot.
            xlabel (str): the x axis label.
            ylabel (str): the y axis label.

        Returns:
            fig (matplotlib figure): figure containing the 6 lines.
        """
        fig = plt.figure()
        plt.plot(x, y1, marker1, label=label1)
        plt.plot(x, y2, marker2, label=label2)
        plt.plot(x, y3, marker3, label=label3)
        plt.plot(x, y4, marker4, label=label4)
        plt.plot(x, y5, marker5, label=label5)
        plt.plot(x, y6, marker6, label=label6)
        plt.title(title, fontweight="bold")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.tight_layout()
        return fig

    def make_two_line_plot(
        self, dist_temp, temp, temp_mod, dist_mod, time_temp, time_mod
    ):
        """
        Creates a 2 line plot.

        Args:
            dist_temp (ndarray): ndarray contains the distance-temperature data.
            temp (ndarray): ndarray containing the temperature data.
            temp_mod (ndarray): ndarray containing the distance modulus data.
            time_temp (ndarray): ndarray containing time-temperature data.
            time_mod (ndarray): ndarray containing the time modulus data.

        Returns:
            fig (matplotlib figure): Figure containing 2 line plot.
        """
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(
            dist_temp,
            np.mean(temp, axis=1),
            "--ko",
            linewidth=1.5,
            markerfacecolor="b",
            markersize=8,
        )
        axs[0].plot(dist_mod, np.mean(temp_mod, axis=1), "r", linewidth=1.5)

        # Xlim - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlim.html
        # Ylim - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylim.html
        mean_temp_axis1 = np.mean(temp, axis=1)
        mean_temp_mod_axis1 = np.mean(temp_mod, axis=1)
        # Set the X and Y limites according to MATLAB code.
        plt.xlim([np.min(dist_temp), np.max(dist_temp)])
        plt.ylim(
            [
                min(mean_temp_axis1.min(), mean_temp_mod_axis1.min()),
                max(mean_temp_axis1.max(), mean_temp_mod_axis1.max()),
            ]
        )

        plt.title("Stream Temperature Along the Reach", fontsize=11, fontweight="bold")
        plt.xlabel("Distance Downstream (m)", fontsize=9)
        plt.ylabel("Temperature (°C)", fontsize=9)
        plt.legend(["Measured", "Modeled"], loc="best")

        # Create matching plot from MATLAB code.
        # Incorrect Graph, coming back for beta
        axs[1].plot(time_temp, np.mean(temp, axis=0), "b", linewidth=1.5)
        axs[1].plot(time_mod, np.mean(temp_mod, axis=0), "r", linewidth=1.5)

        mean_temp = np.mean(temp, axis=0)
        mean_temp_mod = np.mean(temp_mod, axis=0)
        # Set the X and Y limits according to matlab code.
        plt.xlim([np.min(time_temp), np.max(time_temp)])
        plt.ylim(
            [
                min(mean_temp.min(), mean_temp_mod.min()) - 1,
                max(mean_temp.max(), mean_temp_mod.max()) + 1,
            ]
        )

        plt.title("Stream Temperature Over Time", fontsize=11, fontweight="bold")
        plt.xlabel("Time (min)", fontsize=9)
        plt.ylabel("Temperature (°C)", fontsize=9)
        plt.legend(["Measured", "Modeled"], loc="best")

        plt.tight_layout()
        return fig

    def make_bar_charts(self, change):
        """
        Creates a bar chart figure to compare the difference in values between Discharge, GW Temp, VTS, and, "Shade"

        Args:
            change (ndarray): ndarray array containing the data for differences.

        Returns:
            fig (matplotlib figure): figure containing bar chart.
        """
        fig = plt.figure()
        # Bar Chart - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar
        plt.bar(
            ["Discharge", "GW Temp", "VTS", "Shade"],
            change[:, 0],
            label="Decrease Value",
        )
        plt.bar(
            ["Discharge", "GW Temp", "VTS", "Shade"],
            change[:, 1],
            label="Increase Value",
        )
        plt.title(
            "Change in Average Stream Temperature With Changing Variables",
            fontname="Arial",
            fontsize=12,
            fontweight="bold",
        )
        plt.ylabel("Change (°C)", fontname="Arial", fontsize=12, fontweight="bold")
        plt.xlabel(
            "Adjusted Variable", fontname="Arial", fontsize=12, fontweight="bold"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        return fig

    def save_plots(self, *args, path):
        """
        Takes a list of figures and saves them to a pdf.

        Args:
            *args ([figures]): list of figures to save.

        Return:
            None
        """
        pdf_path = os.path.join(os.getcwd(), "Results", "PDFs", f"{path}.pdf")
        print(f"Saving PDF to {pdf_path}...")
        plots_pdf = PdfPages(pdf_path)
        for fig in args:
            plots_pdf.savefig(fig)
        plots_pdf.close()
        plt.close("all")
        print("Done!")
