import os
import sys
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Plotting.plotting_class import Plotting


def create_hflux_errors_plots(
    residuals,
    dist_temp,
    temp,
    temp_mod,
    dist_mod,
    time_temp,
    time_mod,
    return_graphs=False,
):
    """
    Creates and saves a PDF with all hflux error plots.

    Args:
        residuals (ndarray): ndarray containing the data for the heat map.
        dist_temp (ndarray): ndarray contains the distance-temperature data.
        temp (ndarray): ndarray containing the temperature data.
        temp_mod (ndarray): ndarray containing the distance modulus data.
        time_temp (ndarray): ndarray containing time-temperature data.
        time_mod (ndarray): ndarray containing the time modulus data.

    Returns:
        None.
    """
    plc = Plotting()
    fig1 = plc.make_residual_plot(
        residuals,
        "Time (min)",
        "Distance (m)",
        "Modeled Temperature Residuals",
        "Model Residuals (°C)",
        [0, 1400, 0, 1400],
    )
    fig2 = plc.make_two_line_plot(
        dist_temp, temp, temp_mod, dist_mod, time_temp, time_mod
    )
    plc.save_plots(fig1, fig2, path="hflux_errors")
    if return_graphs:
        return fig1, fig2
