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
    """
    Checks for errors related to the heat flux calculations.
        - Checks Value Errors for number of arguments.
        - Checks Type Errors for type/shape of arguments
    Params: *args - list of arguements to be assigned.
    Returns: null
    """

    #Initialize variables. 
    unattend = False
    num_args = len(args)

    #Check if the number of arguments is correct.
    if num_args not in [6, 7]:
        raise ValueError("Invalid number of arguments")
    
    #Assign arguments to variables based on number of arguments.
    if num_args == 7:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, unattend = args
    else:
        time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp = args

    #Check for Boolean, still to be figured out. 
    if unattend:
        print("Checking input arguments...")

    #Type check variables to ensure they are column vectors. 
    if time_mod.ndim != 2 or time_mod.shape[1] != 1:
        raise TypeError("Time_m must be a column vector")
    if dist_mod.ndim != 2 or dist_mod.shape[1] != 1:
        raise TypeError("Dist_m must be a column vector")
    if time_temp.ndim != 2 or time_temp.shape[1] != 1:
        raise TypeError("Time_temp must be a column vector")
    if dist_temp.ndim != 2 or dist_temp.shape[1] != 1:
        raise TypeError("Dist_temp must be a column vector")


