from hflux_shortwave import hflux_shortwave
from hflux_longwave import hflux_longwave
from hflux_bed import hflux_bed
from hflux_latent import hflux_latent
from hflux_sensible import hflux_sensible

# %   hflux_flux calculates the total heat entering and leaving a stream 
# %   for a width of stream section over time in a way that can be called by 
# %   hflux.m.  
# %   The net heat flux is a function of shortwave radiation, 
# %   longwave radiation, evaporation (latent heat), sensible heat, and 
# %   bed conduction.
# %
# % Input:
# %   Note: must all be for the same time period and distance (ie same size)
# %   settings = an array of values specifying solution methods
# %   solar_rad = an array of values for solar radiation (W/m^2) 
# %   air_temp = an array of air temperature values (deg C) 
# %   rel_hum = an array of relative humidity values (unitless)
# %   water_temp = an array of values for stream temperature (deg C)
# %   wind_speed = an array of wind speed values (m/s)
# %   z = elevation of station where met data was obtained (m)
# %   bed_temp=array of bed temperatures
# %   depth_of_meas=distance between stream temp measurement and stream bed
# %   temperature
# %   shade = amount of shade (0-1)
# %   cl = cloud cover (0-1)
# %
# % Output:
# %  [net shortwave longwave atm back land latent sensible bed] = 
# %  heat fluxes to be employed by hflux

def hflux_flux(settings, solar_rad, air_temp, rel_hum, water_temp, wind_speed, z, sed_type, bed_temp, depth_of_meas, shade, vts, cl, sol_refl, WP_m, width_m):
    eq1 = settings[0][0] #Shortwave radiation equation method
    ## 1 = Equation [3] in text, includes correction for reflection
    ## 2 = Equation [4] in text, includes albedo correction

    eq2 = settings[0][1] #Latent Heat Flux Equation
    ## 1 = Equation [14] in text, Penman equation
    ## 2 = Equation [17] in text, mass transfer method

    eq3 = settings[0][2] #Sensible Heat Equation
    ## 1 = Equation [20] in text, Bowen ratio method
    ## 2 = Equation [24] in text, from Dingman, 1994
    
    ### SHORTWAVE RADIATION
    shortwave = hflux_shortwave(solar_rad,shade,sol_refl,eq1)

    ### LONGWAVE RADIATION
    # Calculate longwave radiation
    # Westoff, et al. 2007/Boyd and Kasper, 2003
    longwave, atm_rad, back_rad, land_rad = hflux_longwave(air_temp,water_temp,rel_hum,cl,vts)

    ### LATENT HEAT
    # Calculate energy used for evaporation (latent heat) using the Penman
    # equation for open water
    latent = hflux_latent(water_temp,air_temp,rel_hum,wind_speed,shortwave,longwave,z,eq2)

    ### SENSIBLE HEAT
    # Calculate the sensible heat flux, which is the heat exchange between
    # the water and the air (driven by temperature differences)
    sensible = hflux_sensible(water_temp,air_temp,rel_hum,wind_speed,z,latent,eq3)

    ### STREAMBED CONDUCTION
    # Calculate the heat flux through the streambed
    bed = hflux_bed(sed_type,water_temp,bed_temp,depth_of_meas,width_m,WP_m)

    net = (shortwave + longwave + latent + sensible + bed) * 60
    return net, shortwave, longwave, atm_rad, back_rad, land_rad, latent, sensible, bed 
