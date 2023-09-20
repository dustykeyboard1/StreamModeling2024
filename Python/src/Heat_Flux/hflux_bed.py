import pandas as pd
import numpy as np
import math

# % Inputs:
# % hflux_bed.m calculates the heat flux through the stream bed 
# % Inputs:
# % sed_type = a single value or array of type 'cell' that describe the
# %            sediment type as clay, sand, gravel, or cobbles
# % water_temp: water tempature
# % bed_temp: stream bed temprature measurements
# % depth_of_meas: depth below the stream bed that temperature measurements
# % were collected
# % width_m: width of the stream (meters)
# % WP_m: wetted perimeter of the stream bed (meters)
# % Output:
# % bed: the heat flux through the stream bed

def hflux_bed(sed_type, water_temp, bed_temp, depth_of_measure, width_m, wp_m):
    for i in range(len(sed_type)):
        match sed_type[i]:
            case 1:
                sed_type[i] = .84 #(W/m*C)
            case 2:
                sed_type[i] = 1.2 #(W/m*C)
            case 3:
                sed_type[i] = 1.4 #(W/m*C)
            case 4:
                sed_type[i] = 2.5 #(W/m*C)
    
    return heat_flux_bed(sed_type, water_temp, bed_temp, depth_of_measure, width_m, wp_m)
    
### Heat flux through the stream bed
def heat_flux_bed(sed_type, water_temp, bed_temp, depth_of_measure, width_m, wp_m):
    result = np.empty(0)
    for i in range(len(sed_type)):
        val = (wp_m[i] / width_m[i]) * (-sed_type[i] * ((water_temp[i] - bed_temp[i]) / depth_of_measure[i]))
        result = np.append(result, [val])
    return result
            