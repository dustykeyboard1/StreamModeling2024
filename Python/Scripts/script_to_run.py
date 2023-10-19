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

# Dynamically find and set the root directory.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from Python.src.Utilities.Input_reader import readFromFile
from Python.src.Core.hflux_errors import handle_errors
from Python.src.Core.hflux import hflux
from Python.src.Heat_Flux.hflux_sens import hflux_sens


def script_to_run():
    """
    Script to run the program.
    Params: None
    Returns: None
    """

    # Read in input data from helper funciton.
    filename = os.path.join(os.getcwd(), "Data", "example_data.xlsx")
    input_data = readFromFile(filename)

    # Use helper functions (hflux(), handle_errors() and sens())  to calculate values.
    # Helper functions will also plot and display results.
    temp_mod, matrix_data, node_data, flux_data = hflux(input_data)
    rel_err, me, mae, mse, rmse, nrmse = handle_errors(
        input_data["time_mod"][0],
        input_data["dist_mod"][0],
        temp_mod,
        input_data["temp_t0_data"][0],
        input_data["temp_x0_data"][0],
        input_data["temp"],
    )
    sens = hflux_sens(input_data, [-0.01, 0.01], [-2, 2], [-0.1, 0.1], [-0.1, 0.1])

    # Save output to CSV files using Numpy.
    # np.savetxt() - https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
    path = os.path.join(os.getcwd(), "Results", "CSVs")
    np.savetxt(f"{path}/temp_mod.csv", temp_mod, delimiter=",")
    np.savetxt(f"{path}/temp.csv", input_data["temp"].transpose(), delimiter=",")
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
    script_to_run()
