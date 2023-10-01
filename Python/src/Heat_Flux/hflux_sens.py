"""
Author: Michael Scoleri
Date: 09-26-23
File: test_hflux_errors.py
Functionality: Implementation of hflux_sens.m
"""

import os
import sys
#Dynamically find and set the root directory.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

# from src.Utilities.hflux_format import hflux_format
from src.Core.hflux import hflux

import numpy as np
import gc
import matplotlib as plt

def hflux_sens(input_data,dis_high_low,T_L_high_low,vts_high_low,shade_high_low):
    '''
    Implementation of hflux_sens.m
    Parameters: input_data,dis_high_low,T_L_high_low,vts_high_low,shade_high_low
    Returns: Sens - dictionary
    '''

    print("Calculating high and low values...")

    #Create low and high values for each parameter.
    dis_low = input_data['dis_data'][:, 1:] + dis_high_low[0]
    dis_high = input_data['dis_data'][:, 1:] + dis_high_low[1]
    T_L_low = input_data['T_L_data'][:, 1] + T_L_high_low[0]
    T_L_high = input_data['T_L_data'][:, 1] + T_L_high_low[1]
    vts_low = input_data['shade_data'][:, 2] + vts_high_low[0]
    vts_high = input_data['shade_data'][:, 2] + vts_high_low[1]
    shade_low = input_data['shade_data'][:, 1] + shade_high_low[0]
    shade_high = input_data['shade_data'][:, 1] + shade_high_low[1]

    #Create hflux-ready arrays from the low and high values
    #Use hstack to concatenate along the 2nd axis.
    #Use newaxis to index arrays.
    #np.hstack - https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy-hstack
    #np.newaxis - https://numpy.org/devdocs/reference/constants.html#numpy.newaxis
    dis_data_low = np.hstack((input_data['dis_data'][:, 0][:, np.newaxis], dis_low))
    dis_data_high = np.hstack((input_data['dis_data'][:, 0][:, np.newaxis], dis_high))

    # Initialize T_L_data_low and T_L_data_high with zeros.
    T_L_data_low = np.zeros(input_data['T_L_data'].shape)
    T_L_data_high = np.zeros(input_data['T_L_data'].shape)

    #Set T_L_data_low and T_L_data_high values.
    T_L_data_low[:, 0] = input_data['T_L_data'][:, 0]
    T_L_data_low[:, 1] = T_L_low
    T_L_data_high[:, 0] = input_data['T_L_data'][:, 0]
    T_L_data_high[:, 1] = T_L_high

    # Initialize vts_data_low and vts_data_high with zeros. 
    vts_data_low = np.zeros(input_data['shade_data'].shape)
    vts_data_high = np.zeros(input_data['shade_data'].shape)

    #Set vts_data_low and vts_data_high values.
    vts_data_low[:, 0] = input_data['shade_data'][:, 0]
    vts_data_low[:, 1] = input_data['shade_data'][:, 1]
    vts_data_low[:, 2] = vts_low

    vts_data_high[:, 0] = input_data['shade_data'][:, 0]
    vts_data_high[:, 1] = input_data['shade_data'][:, 1]
    vts_data_high[:, 2] = vts_high

    #Initalize shade_data_low and shade_data_high with zeros. 
    shade_data_low = np.zeros(input_data['shade_data'].shape)
    shade_data_high = np.zeros(input_data['shade_data'].shape)

    shade_data_low[:, 0] = input_data['shade_data'][:, 0]
    shade_data_low[:, 1] = shade_low
    shade_data_low[:, 2] = input_data['shade_data'][:, 2]
    shade_data_high[:, 0] = input_data['shade_data'][:, 0]
    shade_data_high[:, 1] = shade_high
    shade_data_high[:, 2] = input_data['shade_data'][:, 2]

    #Create multiple copies of input_data for and modifying specifc keys. 
    input_data_base = input_data.copy()
    input_data_lowdis = input_data.copy()
    input_data_lowdis['dis_data'] = dis_data_low
    input_data_highdis = input_data.copy()
    input_data_highdis['dis_data'] = dis_data_high
    input_data_lowT_L = input_data.copy()
    input_data_lowT_L['T_L_data'] = T_L_data_low
    input_data_highT_L = input_data.copy()
    input_data_highT_L['T_L_data'] = T_L_data_high
    input_data_lowvts = input_data.copy()
    input_data_lowvts['shade_data'] = vts_data_low
    input_data_highvts = input_data.copy()
    input_data_highvts['shade_data'] = vts_data_high
    input_data_lowshade = input_data.copy()
    input_data_lowshade['shade_data'] = shade_data_low
    input_data_highshade = input_data.copy()
    input_data_highshade['shade_data'] = shade_data_high

    print('...Done!')
    print('     ')
    print("Running HLUX for the base, high, and low cases.")


    #Run hlux.m for middle (base) values, then for high and low values of
    #each parameter with other parameters kept at base values
    temp_mod_base, _, _, _ = hflux(input_data_base)
    temp_mod_lowdis, _, _, _ = hflux(input_data_lowdis)
    temp_mod_highdis, _, _, _ = hflux(input_data_highdis)
    temp_mod_lowT_L, _, _, _ = hflux(input_data_lowT_L)
    temp_mod_highT_L, _, _, _ = hflux(input_data_highT_L)
    temp_mod_lowvts, _, _, _ = hflux(input_data_lowvts)
    temp_mod_highvts, _, _, _ = hflux(input_data_highvts)
    temp_mod_lowshade, _, _, _ = hflux(input_data_lowshade)
    temp_mod_highshade, _, _, _ = hflux(input_data_highshade)

    print('...Done!')
    print('     ')
    print('Writing output data.')

    #Store outputs from hflux to dictionaries.
    base = {'temp': temp_mod_base, 'mean': np.mean(temp_mod_base, axis=1)}
    lowdis = {'temp': temp_mod_lowdis, 'mean': np.mean(temp_mod_lowdis, axis=1)}
    highdis = {'temp': temp_mod_highdis, 'mean': np.mean(temp_mod_highdis, axis=1)}
    lowT_L = {'temp': temp_mod_lowT_L, 'mean': np.mean(temp_mod_lowT_L, axis=1)}
    highT_L = {'temp': temp_mod_highT_L, 'mean': np.mean(temp_mod_highT_L, axis=1)}
    lowvts = {'temp': temp_mod_lowvts, 'mean': np.mean(temp_mod_lowvts, axis=1)}
    highvts = {'temp': temp_mod_highvts, 'mean': np.mean(temp_mod_highvts, axis=1)}
    lowshade = {'temp': temp_mod_lowshade, 'mean': np.mean(temp_mod_lowshade, axis=1)}
    highshade = {'temp': temp_mod_highshade, 'mean': np.mean(temp_mod_highshade, axis=1)}

    # Store structures in dictionary
    sens = {
        'dis_l': dis_data_low,
        'dis_h': dis_data_high,
        'TL_l': T_L_data_low,
        'TL_h': T_L_data_high,
        'vts_l': vts_low,
        'vts_h': vts_high,
        'sh_l': shade_data_low,
        'sh_h': shade_data_high,
        'base': base,
        'lowdis': lowdis,
        'highdis': highdis,
        'lowT_L': lowT_L,
        'highT_L': highT_L,
        'lowvts': lowvts,
        'highvts': highvts,
        'lowshade': lowshade,
        'highshade': highshade
    }

    print("...Done!")
    print("    ")

    #Make sensitivity plots.
    #Following all axes, line, label and function parameters from MATLAB code.
    plt.figure()
    plt.subplot(2,2,1)

    plt.plot(input_data['dist_mod'], sens['lowdis']['mean'], '--b', linewidth=2)
    plt.plot(input_data['dist_mod'], sens['base']['mean'], 'k', linewidth=2)
    plt.plot(input_data['dist_mod'], sens['highdis']['mean'], '--r', linewidth=2)
    #Follow X-Limits set by MATLAB code.
    plt.xlim = ([np.min(input_data['temp_t0_data'][:,0]), 
                 np.max(input_data['temp_t0_data'][:,0])])
    plt.title('Discharge', fontweight = 'bold')
    plt.xlabel('Distance Downstream(m)')
    plt.ylabel('Temperature (°C)')
    plt.legend(['Low', 'Base', 'High'])
    plt.tight_layout()

    plt.figure()
    plt.subplot(2,2,2)
    plt.plot(input_data['dist_mod'],sens['lowT_L']['mean'],'--b',linewidth=2)
    plt.plot(input_data['dist_mod'],sens['base']['mean'],'k',linewidth = 2)
    plt.plot(input_data['dist_mod'],sens['highT_L']['mean'],'--r', linewidth = 2)
    #Follow X-Limits set by MATLAB code.
    plt.xlim = ([np.min(input_data['temp_t0_data'][:,0]), 
                 np.max(input_data['temp_t0_data'][:,0])])
    plt.title('Groundwater Temperature', fontweight='bold')
    plt.ylabel('Temperature (°C)')
    plt.xlabel('Distance Downstream (m)')
    plt.legend(['Low','Base','High'])
    plt.tight_layout()

    plt.figure()
    plt.subplot(2,2,3)
    plt.plot(input_data['dist_mod'],sens['lowvts']['mean'],'--b',linewidth=2)
    plt.plot(input_data['dist_mod'],sens['base']['mean'],'k',linewidth=2)
    plt.plot(input_data['dist_mod'],sens['highvts']['mean'],'--r',linewidth=2)
    #Follow X-Limits set by MATLAB code.
    plt.xlim = ([np.min(input_data['temp_t0_data'][:,0]), 
                 np.max(input_data['temp_t0_data'][:,0])])
    plt.title('View to Sky Coefficient', fontweight="bold")
    plt.xlabel('Distance Downstream (m)')
    plt.ylabel('Temperature (°C)')
    plt.legend(['Low','Base','High'])
    plt.tight_layout()

    plt.figure()
    plt.subplot(2,2,4)
    plt.plot(input_data['dist_mod'],sens['lowshade']['mean'],'--b',linewidth=2)
    plt.plot(input_data['dist_mod'],sens['base']['mean'],'k',linewidth=2)
    plt.plot(input_data['dist_mod'],sens['highshade']['mean'],'--r',linewidth=2)
    plt.xlim = ([np.min(input_data['temp_t0_data'][:,0]), 
                 np.max(input_data['temp_t0_data'][:,0])])
    #Follow X-Limits set by MATLAB code.
    plt.title('Shade', fontweight="bold")
    plt.xlabel('Distance Downstream (m)')
    plt.ylabel('Temperature (°C)')
    plt.legend(['Low','Base','High'])
    plt.tight_layout()

    #Calculate total percent change in stream temperature
    change = np.array([
        [np.mean(sens['lowdis']['temp']) - np.mean(sens['base']['temp']),
        np.mean(sens['highdis']['temp']) - np.mean(sens['base']['temp'])],
        [np.mean(sens['lowT_L']['temp']) - np.mean(sens['base']['temp']),
        np.mean(sens['highT_L']['temp']) - np.mean(sens['base']['temp'])],
        [np.mean(sens['lowvts']['temp']) - np.mean(sens['base']['temp']),
        np.mean(sens['highvts']['temp']) - np.mean(sens['base']['temp'])],
        [np.mean(sens['lowshade']['temp']) - np.mean(sens['base']['temp']),
        np.mean(sens['highshade']['temp']) - np.mean(sens['base']['temp'])]
    ])

    plt.figure()
    #Bar Chart - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar
    plt.bar(['Discharge', 'GW Temp', 'VTS', 'Shade'], change[:, 0], label='Decrease Value')
    plt.bar(['Discharge', 'GW Temp', 'VTS', 'Shade'], change[:, 1], label='Increase Value')
    plt.title('Change in Average Stream Temperature With Changing Variables', fontname='Arial', fontsize=14, fontweight='bold')
    plt.ylabel('Change (°C)', fontname='Arial', fontsize=12, fontweight='bold')
    plt.xlabel('Adjusted Variable', fontname='Arial', fontsize=12, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show(block=False)

    return sens


