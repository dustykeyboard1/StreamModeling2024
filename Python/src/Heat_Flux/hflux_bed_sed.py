import pandas as pd
import numpy as np
from interpolation import interpolation

def hflux_bed_sed(sed_type, dist_bed, dist_mod):

    # Inputs:
    # sed_type = a single value or array of type 'cell' that describe the
    #             sediment type as clay, sand, gravel, or cobbles
    # dist_bed = distances in meters where the sediment type was observed
    # dist_mod = interpolated distances in meters used in the model

    ## we are mapping each of the accepted sediment types to a number
    ## as seen below. To add a new accepted sediment, follow the 
    ## formating as such:
    ## case "SEDIMENT_NAME":
    ##      array[index] = SEDIMENT_NUMBER
    ## clay -> 1
    ## sand -> 2
    ## gravel -> 3
    ## cobbles -> 4
    for index in range(len(sed_type)):
        match sed_type[index].lower():
            case "cobbles":
                sed_type[index] = 1
            case "sand":
                sed_type[index] = 2
            case "clay":
                sed_type[index] = 3
            case "gravel":
                sed_type[index] = 4
            case _:
                print("Invalid sediment:" , sed_type[index] , "detected at index" , index , ".")
                sed_type[index] = -1 # error value, we can change later if we want
    
    print(sed_type)
    return interpolation(dist_bed, sed_type, dist_mod, "nearest")