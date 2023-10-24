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
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#Dynamically find and set the root directory. 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from Python.src.Utilities.Input_reader import readFromFile
from Python.src.Core.hflux_errors import handle_errors
from Python.src.Core.hflux import hflux
from Python.src.Core.heat_flux import HeatFlux
from Python.src.Heat_Flux.hflux_sens import hflux_sens
from Python.src.Utilities.data_table_class import DataTable
def script_to_run():
    '''
    Script to run the program. 
    Params: None
    Returns: None
    '''

    #Read in input data from helper funciton.
    t1 = datetime.now()
    filename = os.path.join(os.getcwd(), 'Data', 'example_data.xlsx')
    data_table = DataTable(filename)
    input_data = data_table.get_input_data(filename)

    heat_flux = HeatFlux(data_table)
    #Use helper functions (hflux(), handle_errors() and sens())  to calculate values.
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
    handle_errors(time_mod, time_temp, temp, temp_dt, temp_mod, dist_temp, dist_mod)
    t3 = datetime.now()
    sens = hflux_sens(input_data, [-0.01, 0.01],[-2, 2],[-0.1, 0.1],[-0.1, 0.1])
    t4 = datetime.now()


    # Save output to CSV files using Numpy.
    # np.savetxt() - https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
    path = os.path.join(os.getcwd(), 'Results', 'CSVs')
    np.savetxt(f"{path}/temp_mod.csv", temp_mod, delimiter=",")
    np.savetxt(f"{path}/temp.csv", data_table.temp, delimiter=",")
    np.savetxt(f"{path}/rel_err.csv", rel_err, delimiter=",")
    np.savetxt(f"{path}/heatflux_data.csv", flux_data['heatflux'], delimiter=",")
    np.savetxt(f"{path}/solarflux_data.csv", flux_data['solarflux'], delimiter=",")
    np.savetxt(f"{path}/solar_refl_data.csv", flux_data['solar_refl'], delimiter=",")
    np.savetxt(f"{path}/long_data.csv", flux_data['long'], delimiter=",")
    np.savetxt(f"{path}/atmflux_data.csv", flux_data['atmflux'], delimiter=",")
    np.savetxt(f"{path}/landflux_data.csv", flux_data['landflux'], delimiter=",")
    np.savetxt(f"{path}/backrad_data.csv", flux_data['backrad'], delimiter=",")
    np.savetxt(f"{path}/evap_data.csv", flux_data['evap'], delimiter=",")
    np.savetxt(f"{path}/sensible_data.csv", flux_data['sensible'], delimiter=",")
    np.savetxt(f"{path}/conduction_data.csv", flux_data['conduction'], delimiter=",")
    t2 = datetime.now()
    print("...Done!")
    print("total time: ", t2-t1)

if __name__ == "__main__":
    script_to_run()