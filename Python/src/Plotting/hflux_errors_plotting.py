import os
import sys
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Plotting.plotting_class import Plotting


def create_hflux_errors_plots(
    residuals, dist_temp, temp, temp_mod, dist_mod, time_temp, time_mod
):
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
