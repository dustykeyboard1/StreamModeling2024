import numpy as np
import math

# Input:
#    Note: must all be for the same time period and distance (ie same size)
#    air_temp = an array of air temperature values (deg C) 
#    rel_hum = an array of relative humidity values (unitless)
#    water_temp = an array of values for stream temperature (deg C)
#    wind_speed = an array of wind speed values (m/s)
#    z = elevation of station where met data was obtained (m)
#    latent = latent heat fulx (W/m^2)
#    eq3 = case1 calculates the sensible heat transfer based on the Bowen ratio heat flux
#          case2 calculates sensible heat transfer based on temperature differences,
#                Dingman, 1994
#                *This switch is set in hflux_flux.m
def hflux_sensible(water_temp, air_temp, rel_hum, wind_speed, z, latent, eq3):
    z = 150
    eq3 = 1
    air_temp = np.array([20, 20, 20, 20])
    rel_hum = np.array([55, 55, 55, 55])
    water_temp = np.array([17.443, 17.443, 17.443, 17.443])
    wind_speed = np.array([0, 0, 0, 0])
    latent = np.array([6.9973, 6.9973, 6.9973, 6.9973])
    
    match eq3:
        case 1:
            ews = saturated_vp(water_temp)
            ewa = actual_vp(rel_hum, ews)
            air_pressure = 101.3 * (((293 - (.0065 * z)) / 293) ** 5.256)
            br = bowens_ratio(air_pressure, water_temp, air_temp, ews, ewa)
            sensible = case1_sensible(br, latent)
        case 2:
            c_air = 1004 # heat-capacity of the air (J/kg deg C)
            rho_air = 1.2041 # density of air at 20 degree C (kg/m^3)
            z_veg = .25 #height of vegetation around stream (m)
            z_met = .5 #height of wind speed measurements (m)

            zd = .7 * z_veg # zero-plane displacement (m)
            z0 = .1 * z_veg # roughness height (m)
            dh_dm = 1 #ratio of diffusivity of heat to diffusivity of momentum
            k = .4 #dimensionless constant

            kh = dh_dm * c_air * rho_air * (k ** 2 / ((math.log((z_met - zd) / z0)) ** 2))
            sensible = case2_sensible(kh, wind_speed, water_temp, air_temp)

    return sensible

### Saturated vapor pressire using stream temp
def saturated_vp(water_temp):
    result = np.empty(0)
    for x in water_temp:
        val = .61275 * math.exp((17.27 * x) / (237.3 + x))
        result = np.append(result, [val])
    return result

### Actual vapor pressure
def actual_vp(rel_hum, ews):
    result = np.empty(0)
    for i in range(len(rel_hum)):
        val = (rel_hum[i] / 100) * ews[i]
        result = np.append(result, [val])
    return result

### Bowen's ratio calculation
def bowens_ratio(air_pressure, water_temp, air_temp, ews, ewa):
    result = np.empty(0)
    for i in range(len(water_temp)):
        val = .00061 * air_pressure * ((water_temp[i] - air_temp[i]) / (ews[i] - ewa[i]))
        result = np.append(result, [val])
    return result

### Sensible case1 calulation
def case1_sensible(br, latent):
    result = np.empty(0)
    for i in range(len(latent)):
        val = br[i] * latent[i]
        result = np.append(result, [val])
    return result

### Sensible case2 calulation
def case2_sensible(kh, wind_speed, water_temp, air_temp):
    result = np.empty(0)
    for i in range(len(wind_speed)):
        val = -kh * wind_speed[i] * (water_temp[i] - air_temp[i])
        result = np.append(result, [val])
    return result

hflux_sensible([], [], [], [], [], [], [])