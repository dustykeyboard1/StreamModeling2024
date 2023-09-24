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

input_data = readFromFile('Python/Data/example_data.xlsx')

rel_err, me, mae, mse, rmse, nrmse = handle_errors(input_data['time_mod'], input_data['dist_mod'], temp_mod, input_data['temp_t0_data'][:, 0], 
                                                   input_data['temp_x0_data'][:, 0], input_data['temp'])