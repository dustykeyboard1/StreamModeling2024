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

def hflux_latent(shortwave, longwave, rel_hum, water_temp, wind_speed, air_temp, z, eq2):
    c_air = 1004 # heat capacity of the air (J/kg deg C)
    rho_water = 1000 # Density of water (kg/m^3)
    rho_air = 1.2041 # Density of air at 20 deg C (kg/m^3)
    air_pressure = 101.3 - (.0105 * z) # air pressire
    psy_constant = (c_air * air_pressure) / (.622 * 2450000) # psychometric constant (kPa/deg C)
    #wind function coefficients
    b0 = 1.505E-8; #1/(m/s*kPa)
    b1 = 1.6E-8; #1/kPa

    match eq2:
        case 1:
            e_s = saturation_vp_air(air_temp) 
            e_a = actual_vp(rel_hum, e_s) 
            l_e = latent_heat_vapor(water_temp) 
            s = saturation_vp_slope(air_temp, e_s) 
            r_a = aero_resistance(wind_speed)
            penman_evap = penman_evap_eq(s, shortwave, longwave, rho_water, l_e, psy_constant, c_air, rho_air, e_s, e_a, r_a)
            latent = case1_latent_heat_flux(rho_water, l_e, penman_evap)
        case 2:
            e_s = saturation_vp_air(air_temp) 
            e_a = actual_vp(rel_hum, e_s) 
            ews = saturated_vp_water(water_temp)
            l_e = latent_heat_vapor(water_temp) 
            fw = wind_speed(b0, b1, wind_speed)
            mass_transfer_evap = mass_transfer_evap_eq(fw, ews, e_a)
            latent = case2_latent_heat_flux(rho_water, l_e, mass_transfer_evap)

    return latent

### The saturation vapor pressure equation (air)
def saturation_vp_air(temp):
    result = np.empty(len(temp))
    for i in range(len(temp)):
        result[i] = .611 * math.exp((17.27 * temp[i]) / (237.2 + temp[i]))
    return result

### The actual vapor pressure equation
def actual_vp(rel_hum, e_s):
    result = np.zeros(len(rel_hum))
    for i in range(len(rel_hum)):
        result[i] = (rel_hum[i] / 100.0) * e_s[i]
    return result

### Saturated vapor pressure at the stream surface
def saturated_vp_water(water_temp):
    result = np.zeros(len(water_temp))
    for i in range(len(water_temp)):
        result[i] = .611 * math.exp((17.27 * water_temp[i]) / (237.3 + water_temp[i]))
    return result

### Latent heat of vaporization
def latent_heat_vapor(water_temp):
    result = np.zeros(len(water_temp))
    for i in range(len(water_temp)):
        result[i] = 1000000 * (2.501 - (.002361 * water_temp[i]))
    return result

### Slope of the saturated vapor pressure curve at a given air temperature
def saturation_vp_slope(air_temp, e_s):
    result = np.zeros(len(e_s))
    for i in range(len(e_s)):
        result[i] = (4100 * e_s[i]) / ((237 + air_temp[i]) ** 2)
    return result

### Aerodynamic resistance
def aero_resistance(wind_speed):
    result = np.zeros(len(wind_speed))
    for i in range(len(wind_speed)):
        result[i] = 245 / ((.54 * wind_speed[i]) + .5)
    return result

### Penman evap equation
def penman_evap_eq(s, shortwave, longwave, rho_water, l_e, psy_constant, c_air, rho_air, e_s, e_a, r_a):
    result = np.zeros(len(shortwave))
    for i in range(len(shortwave)):
        result[i] = (((s[i] * (shortwave[i] + longwave[i])) / (rho_water * l_e[i] * (s[i] + psy_constant)))) + ((c_air * rho_air * psy_constant * (e_s[i] - e_a[i])) / (rho_water * l_e[i] * r_a[i] * (s[i] + psy_constant)))
    return result

### Latent heat flux
def case1_latent_heat_flux(rho_water, l_e, penman_evap):
    result = np.zeros(len(l_e))
    for i in range(len(l_e)):
        result[i] = (-rho_water) * l_e[i] * penman_evap[i]
    return result

### Wind speed function
def wind_speed(b0, b1, wind_speed):
    result = np.zeros(len(wind_speed))
    for i in range(len(wind_speed)):
        result[i] = b0 + (b1 * wind_speed[i])

    return result

### Mass transfer evap
def mass_transfer_evap_eq(fw, ews, e_a):
    result = np.zeros(len(ews))
    for i in range(len(ews)):
        result[i] = fw[i] * (ews[i] - e_a[i])
    return result

### Case 2 Latent heat flux
def case2_latent_heat_flux(rho_water, l_e, mass_transfer_evap):
    result = np.zeros(len(l_e))
    for i in range(len(l_e)):
        result[i] = -rho_water * l_e[i] * mass_transfer_evap[i]
    return result