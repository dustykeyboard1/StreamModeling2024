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
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities.interpolation import interpolation_func

def handle_errors(*args):
    """
    Checks for errors related to the heat flux calculations.
        - Checks Value Errors for number of arguments.
        - Checks Type Errors for type/shape of arguments
    Params: *args - list of arguements to be assigned.
    Returns: rel_err, me, mae, mse, rmse, nrmse
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
        raise TypeError("Time_mod must be a column vector.")
    if dist_mod.ndim != 2 or dist_mod.shape[1] != 1:
        raise TypeError("Dist_m must be a column vector.")
    if time_temp.ndim != 2 or time_temp.shape[1] != 1:
        raise TypeError("Time_temp must be a column vector.")
    if dist_temp.ndim != 2 or dist_temp.shape[1] != 1:
        raise TypeError("Dist_temp must be a column vector.")

    #Isinstance - https://docs.python.org/3/library/functions.html#isinstance
    if not isinstance(temp_mod, np.ndarray): 
        raise TypeError("Temp_mod must be a numpy array representing a matrix.")
    if not isinstance(temp, np.ndarray):
        raise TypeError("Temp must be a numpy array representing a matrix.")
    
    if unattend:
        print('...Done!')
        print('    ')
        print('Resampling data to original spatial and temporal resolution...')

    result_list = []
    for i in range(len(time_mod)):
        result = interpolation_func(dist_mod[:,0], temp_mod[:,i], dist_temp[:,0], 'linear')
        result_list.append(result)
    temp_dx = np.array(result_list)

    result_list = []
    for i in range(len(dist_temp)):
        result = interpolation_func(time_mod[:,0], temp_dx[i,:], time_temp[:,0], 'linear')
        result_list.append(result)
    temp_dt = np.array(result_list)

    if unattend:
        print('...Done!')
        print('    ')
        print('Calculating error metrics...')


    #Start Calculating error metrics

    #Percent Relative Error
    rel_err = ((temp - temp_dt) / temp) * 100

    #Mean Residual Error
    me = np.sum(temp - temp_dt) / np.size(temp)

    #Mean absolute Resiudal Error
    mae = np.sum(np.abs(temp - temp_dt)) / np.size(temp)

    #Mean Squared Error
    mse = np.sum((temp - temp_dt)**2) / np.size(temp)

    #Root Mean Squared Error
    rmse = np.sqrt(np.sum((temp - temp_dt)**2) / np.size(temp))

    #Normalized Root Mean Square 
    nrmse = (rmse / (np.max(temp) - np.min(temp)))*100

    if unattend:
        print('...Done!')
        print('     ')

    if unattend: 
        plt.figure()
        residuals = temp - temp_dt
        plt.imshow(residuals)

        plt.figure()
        plt.subplot(2, 1, 1)

        plt.plot(dist_temp[:, 0], np.mean(temp, axis = 1))
        plt.plot(dist_mod, np.mean(temp_mod, axis = 1))

        plt.subplot(2,1, 2)
        plt.plot(dist_temp[:, 0], np.mean(temp, axis=1))
        plt.plot(dist_mod, np.mean(temp_mod, axis=1))

        plt.subplot(2, 1, 2)
        plt.plot(time_temp[:, 0], np.mean(temp, axis=0))
        plt.plot(time_mod, np.mean(temp_mod, axis=0))

        plt.show()

    return rel_err, me, mae, mse, rmse, nrmse 

                 


