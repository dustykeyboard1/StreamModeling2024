import pandas as pd
import numpy as np
import math


# Input:
# %   Note: must all be for the same time period and distance (ie same size)
# %   air_temp = an array of air temperature values (deg C) 
# %   rel_hum = an array of relative humidity values (unitless)
# %   water_temp = an array of values for stream temperature (deg C)
# %   vts = view to sky coefficient (0-1)
# %   cl = cloud cover (0-1)

def hflux_longwave(air_temp, water_temp, rel_hum, cl, vts):
    ### Stefan-Boltzman constant
    s_b = 5.67E-8

    e_s = saturation_vp(air_temp)
    e_a = actual_vp(rel_hum, e_s)
    e_atm = emissivity_atm(e_a, air_temp, cl)
    atm_rad = atmospheric_radiation(e_atm, vts, s_b, air_temp)
    back_rad = back_radiation(s_b, water_temp)
    land_rad = land_radiation(vts, s_b, air_temp)
    longwave = longwave_radiation(atm_rad, back_rad, land_rad)

    return longwave, atm_rad, back_rad, land_rad 

### The saturation vapor pressure equation
def saturation_vp(air_temp):
    result = np.zeros(len(air_temp))
    for i in range(len(air_temp)):
        result[i] = .611 * math.exp((17.27 * air_temp[i]) / (237.2 + air_temp[i]))
    return result

### The actual vapor pressure equation
def actual_vp(rel_hum, e_s):
    result = np.zeros(len(rel_hum))
    for i in range(len(rel_hum)):
        result[i] = (rel_hum[i] / 100) * e_s[i]
    return result

### Emissivity of atmosphere
def emissivity_atm(e_a, air_temp, cl):
    result = np.zeros(len(e_a))
    for i in range(len(e_a)):
        result[i] = 1.72 * ((e_a[i] / (air_temp[i] + 273.2)) ** (1/7)) * (1 + .22 * cl[i] ** 2)
    return result

### Atmospheric Radiation
def atmospheric_radiation(e_atm, vts, s_b, air_temp):
    result = np.zeros(len(e_atm))
    for i in range(len(e_atm)):
        result[i] = .96 * e_atm[i] * vts[i] * s_b * ((air_temp[i] + 273.2) ** 4)
    return result

### Back Radiation from Water Column
def back_radiation(s_b, water_temp):
    result = np.zeros(len(water_temp))
    for i in range(len(water_temp)):
        result[i] = -.96 * s_b * ((water_temp[i] + 273.2) ** 4)
    return result

### Land Cover Radiation
def land_radiation(vts, s_b, air_temp):
    result = np.zeros(len(vts))
    for i in range(len(vts)):
        result[i] = .96 * (1 - vts[i]) * .96 * s_b * ((air_temp[i] + 273.2) ** 4)
    return result

### Longwave Radiation
def longwave_radiation(atm_rad, back_rad, land_rad):
    result = np.zeros(len(atm_rad))
    for i in range(len(atm_rad)):
        result[i] = atm_rad[i] + back_rad[i] + land_rad[i]
    return result