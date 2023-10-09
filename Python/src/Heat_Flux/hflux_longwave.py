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

    e_s = np.vectorize(saturation_vp)(air_temp)
    e_a = (rel_hum / 100) * e_s
    e_atm = 1.72 * ((e_a / (air_temp + 273.2)) ** (1/7)) * (1 + .22 * cl ** 2)
    atm_rad = .96 * e_atm * vts * s_b * ((air_temp + 273.2) ** 4)

    back_rad = -.96 * s_b * ((water_temp + 273.2) ** 4)
    land_rad = .96 * (1 - vts) * .96 * s_b * ((air_temp + 273.2) ** 4)
    longwave = atm_rad + back_rad + land_rad

    return longwave, atm_rad, back_rad, land_rad 

### The saturation vapor pressure equation
def saturation_vp(air_temp_value):
    return .611 * math.exp((17.27 * air_temp_value) / (237.2 + air_temp_value))