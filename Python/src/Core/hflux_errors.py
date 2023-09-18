'''
Author: Michael Scoleri
Date: 09-18-23
File: hflux_errors.py
Functionality: 
- To handle and report errors related to the heat flux calculations.
    - Value Errors, Type Errors, etc. 
'''

import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def handle_errors(*args):

    unattend = False
    num_args = len(args)

    if num_args not in [6, 7]:
        raise ValueError("Invalid number of arguments")
    
    if num_args == 7:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, unattend = args
    else:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp = args

    if unattend:
        print("Checking input arguments...")

    if time_mod.ndim != 2 or time_mod.shape[1] != 1:
        raise TypeError("Time_m must be a column vector")
    if dist_mod.ndim != 2 or dist_mod.shape[1] != 1:
        raise TypeError("Dist_m must be a column vector")
    if time_temp.ndim != 2 or time_temp.shape[1] != 1:
        raise TypeError("Time_temp must be a column vector")
    if dist_temp.ndim != 2 or dist_temp.shape[1] != 1:
        raise TypeError("Dist_temp must be a column vector")


