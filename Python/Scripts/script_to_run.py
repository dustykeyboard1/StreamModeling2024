"""
Author: Michael Scoleri
Date: 09-22-23
File: script_to_run.py
Functionality: Coordinate the entire program
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Dynamically find and set the root directory.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from Python.src.Core.hflux_errors import handle_errors
from Python.src.Core.heat_flux import HeatFlux
from Python.src.Heat_Flux.hflux_sens import HfluxSens
from Python.src.Utilities.data_table_class import DataTable
from Python.src.Plotting.hflux_errors_plotting import create_hflux_errors_plots
from Python.src.Heat_Flux.hflux_sens import HfluxSens


def script_to_run():
    """
    Script to run the program.
    Params: None
    Returns: None
    """

    # Read in input data from helper funciton.
    filename = os.path.join(os.getcwd(), "Data", "example_data.xlsx")
    data_table = DataTable(filename)

    heat_flux = HeatFlux(data_table)
    # Use helper functions (hflux(), handle_errors() and sens())  to calculate values.
    # Helper functions will also plot and display results.
    temp_mod, matrix_data, node_data, flux_data = heat_flux.crank_nicolson_method()
    temp_dt = heat_flux.calculate_temp_dt(temp_mod)

    temp = data_table.temp.transpose()
    rel_err = heat_flux.calculate_percent_relative_error(temp, temp_dt)
    me = heat_flux.calculate_mean_residual_error(temp, temp_dt)
    mae = heat_flux.calculate_mean_absolute_residual_error(temp, temp_dt)
    mse = heat_flux.calculate_mean_squared_error(temp, temp_dt)
    rmse = heat_flux.calculate_root_mean_squared_error(temp, temp_dt)
    nrmse = heat_flux.calculate_normalized_root_mean_square(rmse, temp)

    dist_temp = data_table.dist_temp
    dist_mod = data_table.dist_mod
    time_temp = data_table.time_temp
    time_mod = data_table.time_mod

    heat_flux.create_hlux_plots(temp_mod, flux_data)
    create_hflux_errors_plots(
        (temp - temp_dt), dist_temp, temp, temp_mod, dist_mod, time_temp, time_mod
    )
    hflux_sens = HfluxSens(root_dir)
    high_low_dict = hflux_sens.hflux_sens(
        data_table, [-0.01, 0.01], [-2, 2], [-0.1, 0.1], [-0.1, 0.1]
    )

    sens = hflux_sens.create_new_results(temp_mod, high_low_dict)
    hflux_sens.make_sens_plots(data_table, sens)

    # Save output to CSV files using Numpy.
    # np.savetxt() - https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
    path = os.path.join(os.getcwd(), "Results", "CSVs")
    np.savetxt(f"{path}/temp_mod.csv", temp_mod, delimiter=",")
    np.savetxt(f"{path}/temp.csv", data_table.temp, delimiter=",")
    np.savetxt(f"{path}/rel_err.csv", rel_err, delimiter=",")
    np.savetxt(f"{path}/heatflux_data.csv", flux_data["heatflux"], delimiter=",")
    np.savetxt(f"{path}/solarflux_data.csv", flux_data["solarflux"], delimiter=",")
    np.savetxt(f"{path}/solar_refl_data.csv", flux_data["solar_refl"], delimiter=",")
    np.savetxt(f"{path}/long_data.csv", flux_data["long"], delimiter=",")
    np.savetxt(f"{path}/atmflux_data.csv", flux_data["atmflux"], delimiter=",")
    np.savetxt(f"{path}/landflux_data.csv", flux_data["landflux"], delimiter=",")
    np.savetxt(f"{path}/backrad_data.csv", flux_data["backrad"], delimiter=",")
    np.savetxt(f"{path}/evap_data.csv", flux_data["evap"], delimiter=",")
    np.savetxt(f"{path}/sensible_data.csv", flux_data["sensible"], delimiter=",")
    np.savetxt(f"{path}/conduction_data.csv", flux_data["conduction"], delimiter=",")

    print("...Done!")


if __name__ == "__main__":
    start_time = time.time()
    script_to_run()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime} seconds")
