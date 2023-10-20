"""
Author: Michael Scoleri
File: Plotting_class.py
Date: 10-19-2023
Functionality: Construct a plotting class for plotting capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


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

    def make_basic_plot(
        self,
        x,
        y,
        title,
        xlabel,
        ylabel,
        xlimit,
        ylimit,
        linewidth=None,
        marker=None,
        legend=None,
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

        Returns:
            Figure: Figure of the residual plot.
        """

        pass

        def make_residual_plot(
            self, x, y, xlabel, ylabel, title, colorbar_label, extent
        ):
            """
            Creates a residual plot (heat map).

            Args:
                x (ndarray): The x-axis data.
                y (ndarray): The y-axis data.
                xlabel (str): The label for the x-axis.
                ylabel (str): The label for the y-axis.
                title (str): The title of the graph.
                colorbar_label (str): The label for the colorbar.
                extent (list[float]): List of floats to set the extent of the graph.

            Returns:
                Figure: Figure of line plot.
            """

        pass
