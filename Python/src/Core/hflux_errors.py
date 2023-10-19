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


def handle_errors(*args):
    """
    Checks for errors related to the heat flux calculations.
        - Checks Value Errors for number of arguments.
        - Checks Type Errors for type/shape of arguments
    Params: *args - list of arguements to be assigned.
    Returns: rel_err, me, mae, mse, rmse, nrmse
    """

    # Initialize variables.
    unattend = False
    num_args = len(args)

    # Check if the number of arguments is correct.
    if num_args not in [6, 7]:
        raise ValueError("Invalid number of arguments")

    # Assign arguments to variables based on number of arguments.
    if num_args == 7:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, unattend = args
    else:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp = args

    # Check for Boolean.
    if not unattend:
        print("Checking input arguments...")

    # Type check variables to ensure they are column vectors by checking number of dimnesions and shape.
    # ndim - https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
    # shape - https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    if time_mod.ndim != 1:
        raise TypeError("Time_m must be a column vector")
    if dist_mod.ndim != 1:
        raise TypeError("Dist_m must be a column vector")
    if time_temp.ndim != 1:
        raise TypeError("Time_temp must be a column vector")
    if dist_temp.ndim != 1:
        raise TypeError("Dist_temp must be a column vector.")

    # Ensure temp_mod and temp are ndarrays.
    # Isinstance - https://docs.python.org/3/library/functions.html#isinstance
    if not isinstance(temp_mod, np.ndarray):
        raise TypeError("Temp_mod must be a numpy array representing a matrix.")
    if not isinstance(temp, np.ndarray):
        raise TypeError("Temp must be a numpy array representing a matrix.")

    if not unattend:
        print("...Done!")
        print("    ")
        print("Resampling data to original spatial and temporal resolution...")

    # Performs linear interpolation using dist_mod, temp_mod and dist_temp at each time step.
    # Stores in temp_dx.
    result_list = []
    for i in range(len(time_mod)):
        result = interpolation(dist_mod, temp_mod[:, i], dist_temp)
        result_list.append(result)
    temp_dx = np.array(result_list).transpose()

    # Performs linear interpolation using time_mod, temp_dx, time_temp at each time step.
    # Stores in temp_dt.
    result_list = []
    for i in range(len(dist_temp)):
        result = interpolation(time_mod, temp_dx[i, :], time_temp)
        result_list.append(result)
    temp_dt = np.array(result_list)

    if not unattend:
        print("...Done!")
        print("    ")
        print("Calculating error metrics...")

    # Start Calculating error metrics according to MATLAB code...

    # Percent Relative Error
    temp = temp.transpose()
    rel_err = ((temp - temp_dt) / temp) * 100

    # Mean Residual Error
    me = np.sum(temp - temp_dt) / np.size(temp)

    # Mean absolute Resiudal Error
    mae = np.sum(np.abs(temp - temp_dt)) / np.size(temp)

    # Mean Squared Error
    mse = np.sum((temp - temp_dt) ** 2) / np.size(temp)

    # Root Mean Squared Error
    rmse = np.sqrt(np.sum((temp - temp_dt) ** 2) / np.size(temp))

    # Normalized Root Mean Square
    nrmse = (rmse / (np.max(temp) - np.min(temp))) * 100

    if not unattend:
        print("...Done!")
        print("     ")

    # Begin plotting...
    # Set all labels/titles, font parametets, and axis ratio limits according to MATLAB code.
    if not unattend:
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
        # Incorrect Graph, coming back for beta
        plt.plot(dist_temp, np.mean(temp, axis=1), "b", linewidth=1.5)
        plt.plot(dist_mod, np.mean(temp_mod, axis=1), "r", linewidth=1.5)

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

    return rel_err, me, mae, mse, rmse, nrmse
