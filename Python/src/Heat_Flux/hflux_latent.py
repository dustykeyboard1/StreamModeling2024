import pandas as pd
import numpy as np
import math

# % Input:
# %   Note: must all be for the same time period and distance (ie same size)
# %   shortwave = an array of values for solar radiation (W/m^2) 
# %   longwave = an array of values for longwave radiation (W/m^2) 
# %   rel_hum = an array of relative humidity values (unitless)
# %   water_temp = an array of values for stream temperature (deg C)
# %   wind_speed = an array of wind speed values (m/s)
# %   z = elevation of station where met data was obtained (m)
# %   eq2 = case1 uses the Penman Method to calculate latent heat flux
# %         case2 uses the Mass Transfer Method to calculate latent heat flux
# %               *This switch is set in hflux_flux.m

def hflux_latent(water_temp, air_temp, rel_hum, wind_speed, shortwave, longwave, z, eq2):
    c_air = 1004 # heat capacity of the air (J/kg deg C)
    rho_water = 1000 # Density of water (kg/m^3)
    rho_air = 1.2041 # Density of air at 20 deg C (kg/m^3)
    air_pressure = 101.3 - (.0105 * z) # air pressire
    psy_constant = (c_air * air_pressure) / (.622 * 2450000) # psychometric constant (kPa/deg C)
    #wind function coefficients
    b0 = 1.505E-8; #1/(m/s*kPa)
    b1 = 1.6E-8; #1/kPa

    e_s = np.vectorize(saturation_vp_air)(air_temp) 
    e_a = (rel_hum / 100.0) * e_s
    l_e = 1000000 * (2.501 - (.002361 * water_temp))

    match eq2:
        case 1:
            s = (4100 * e_s) / ((237 + air_temp) ** 2)
            r_a = 245 / ((.54 * wind_speed) + .5)
            penman_evap = (((s * (shortwave + longwave)) / (rho_water * l_e * (s + psy_constant)))) + ((c_air * rho_air * psy_constant * (e_s - e_a)) / (rho_water * l_e * r_a * (s + psy_constant)))
            latent = (-rho_water) * l_e * penman_evap
        case 2: 
            ews = np.vectorize(saturation_vp_water)(water_temp)
            fw = b0 + (b1 * wind_speed)
            mass_transfer_evap = fw * (ews - e_a)
            latent = -rho_water * l_e * mass_transfer_evap

    return latent

### The saturation vapor pressure equation (air)
def saturation_vp_air(air_temp_value):
    return .611 * math.exp((17.27 * air_temp_value) / (237.2 + air_temp_value))

### Saturated vapor pressure at the stream surface
def saturation_vp_water(water_temp_value):
    return .611 * math.exp((17.27 * water_temp_value) / (237.3 + water_temp_value))