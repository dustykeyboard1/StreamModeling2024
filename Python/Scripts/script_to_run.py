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
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
print(root_dir)

from Python.src.Utilities.Input_reader import readFromFile
from Python.src.Core.hflux_errors import handle_errors
from Python.src.Core.hflux import hflux
from Python.src.Utilities import sens
input_data = readFromFile(root_dir + '/Python/Data/example_data.xlsx')

# temp_mod, matrix_data, node_data, flux_data = hflux()
# rel_err, me, mae, mse, rmse, nrmse = handle_errors(input_data['time_mod'], input_data['dist_mod'], temp_mod, input_data['temp_t0_data'][:, 0], 
                                                #    input_data['temp_x0_data'][:, 0], input_data['temp'])

# Create mock data for testing
time_mod = np.array([[1], [2], [3]])
dist_mod = np.array([[1], [2], [3]])
temp_mod = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

time_temp = np.array([[1], [2], [3]])
dist_temp = np.array([[1], [2], [3]])
temp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

unattend = True 

# Call the handle_errors function
rel_err, me, mae, mse, rmse, nrmse = handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, unattend)
sens = sens(input_data, [-0.01, 0.01],[-2, 2],[-0.1, 0.1],[-0.1, 0.1])

# Save output to CSV files
# np.savetxt() - https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
path = "Python/Results/CSVs/"
np.savetxt(f"{path}temp_mod.csv", temp_mod, delimiter=",")
np.savetxt(f"{path}temp.csv", temp, delimiter=",")
np.savetxt(f"{path}rel_err.csv", rel_err, delimiter=",")
np.savetxt(f"{path}heatflux_data.csv", flux_data['heatflux'], delimiter=",")
np.savetxt(f"{path}solarflux_data.csv", flux_data['solarflux'], delimiter=",")
np.savetxt(f"{path}solar_refl_data.csv", flux_data['solar_refl'], delimiter=",")
np.savetxt(f"{path}long_data.csv", flux_data['long'], delimiter=",")
np.savetxt(f"{path}atmflux_data.csv", flux_data['atmflux'], delimiter=",")
np.savetxt(f"{path}landflux_data.csv", flux_data['landflux'], delimiter=",")
np.savetxt(f"{path}backrad_data.csv", flux_data['backrad'], delimiter=",")
np.savetxt(f"{path}evap_data.csv", flux_data['evap'], delimiter=",")
np.savetxt(f"{path}sensible_data.csv", flux_data['sensible'], delimiter=",")
np.savetxt(f"{path}conduction_data.csv", flux_data['conduction'], delimiter=",")
