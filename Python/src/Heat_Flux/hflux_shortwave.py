import numpy as np

# Inputs:
#   solar_rad = total incoming solar radiation data at each time_met
#       (program interpolates in time)
#   shade = values for shading (0 to 1, with 0 being min shading and 1 being max shading)
#   sol_refl = portion of solar radiation that is reflected off the surface of the stream
#   eq1 = case1 uses the Ouellet, et al. 2012 and Boyd and Kasper, 2003 methods
#         case2 uses the Magnusson, et al. 2012 nethod
#               *This switch was set in hflux_flux.m
# Outputs:
# shortwave = shortwave radiation (W/m^2)

### Global constant to be used throughout
ALBEDO = 0.05

def hflux_shortwave(solar_rad, shade, sol_refl, eq1):
    match eq1:
        case 1:
            sol_in = (1 - shade) * solar_rad
            shortwave = sol_in - (sol_refl * sol_in)
        case 2:
            shortwave = (1 - ALBEDO) * ((1 - shade) * solar_rad)
    return shortwave