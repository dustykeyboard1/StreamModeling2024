"""
Author: Michael Scoleri, Violet Shi, James Gallagher
Date: 09-18-23
File: hflux_errors.py
Functionality: 
- To handle and report errors related to the heat flux calculations.
    - Value Errors, Type Errors, etc. 
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Dynamically find and set the root directory.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from src.Utilities.interpolation import interpolation


def handle_errors(time_mod, time_temp, temp, temp_dt, temp_mod, dist_temp, dist_mod):
    """
    Checks for errors related to the heat flux calculations.
        - Checks Value Errors for number of arguments.
        - Checks Type Errors for type/shape of arguments

    Args:
    *args (list): list of 6 or 7 arguements to be assigned in one of these 2 orders:
                1: time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, output_suppression
                2: time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp

    Return:
        rel_err (integer): relative error,
        me (float): mean residual error,
        mae (float): mean absoulute residual error,
        mse (float): mean squared error,
        rmse (float): Root Mean Squared Error
        nrmse (float): Normalized Root Mean Square
    """

    # Initialize variables.
    output_suppression = False

    # Begin plotting...
    # Set all labels/titles, font parametets, and axis ratio limits according to MATLAB code.
    if not output_suppression:
        _, ax = plt.subplots()
        residuals = temp - temp_dt
        # Create 2D image from 2D ndarray.
        # Matplotlib Imshow - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        plt.imshow(
            residuals,
            aspect="auto",
            cmap="jet",
            origin="lower",
            extent=[0, 1400, 0, 1400],
        )
        # Matplotlib Color Bar - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        plt.colorbar(label="Model Residuals (°C)")
        plt.axis("square")
        plt.title("Modeled Temperature Residuals", fontsize=11, fontweight="bold")
        plt.xlabel("Time (min)", fontsize=9, fontweight="bold")
        plt.ylabel("Distance (m)", fontsize=9, fontweight="bold")
        ax.invert_yaxis()

        plt.figure()
        plt.subplot(2, 1, 1)
        # Create matching plot from MATLAB code.
        plt.plot(
            dist_temp,
            np.mean(temp, axis=1),
            "--ko",
            linewidth=1.5,
            markerfacecolor="b",
            markersize=8,
        )
        plt.plot(dist_mod, np.mean(temp_mod, axis=1), "r", linewidth=1.5)

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

        plt.subplot(2, 1, 2)
        # Create matching plot from MATLAB code.
        plt.plot(time_temp, np.mean(temp, axis=0), "b", linewidth=1.5)
        plt.plot(time_mod, np.mean(temp_mod, axis=0), "r", linewidth=1.5)

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

        pdf_path = os.path.join(os.getcwd(), "Results", "PDFs", "hflux_errors.pdf")
        plots_pdf = PdfPages(pdf_path)

        # CITE: https://www.geeksforgeeks.org/save-multiple-matplotlib-figures-in-single-pdf-file-using-python/
        # get_fignums Return list of existing
        # figure numbers
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        # iterating over the numbers in list
        for fig in figs:
            # and saving the files
            fig.savefig(plots_pdf, format="pdf")

        plots_pdf.close()
        plt.close("all")

    # return rel_err, me, mae, mse, rmse, nrmse
