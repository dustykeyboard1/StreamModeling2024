import numpy as np
import pandas as pd
import math
import scipy
from .shortwave_reflection_calculations import ShortwaveReflectionCalculations

class HeatFluxCalculations:
    """
    Class that calculates the total heat entering and leaving a stream
    for a width of stream section over time in a way that can be called by
    hflux.m.
    The net heat flux is a function of shortwave radiation,
    longwave radiation, evaporation (latent heat), sensible heat, and
    bed conduction.
    """

    def _hflux_shortwave(self, solar_rad, shade, sol_refl, eq1):
        """
        Inputs:
            solar_rad = total incoming solar radiation data at each time_met
                (program interpolates in time)
            shade = values for shading (0 to 1, with 0 being min shading and 1 being max shading)
            sol_refl = portion of solar radiation that is reflected off the surface of the stream
            eq1 = case1 uses the Ouellet, et al. 2012 and Boyd and Kasper, 2003 methods
                case2 uses the Magnusson, et al. 2012 nethod
                *This switch was set in hflux_flux.m
        Outputs:
            shortwave = shortwave radiation (W/m^2)
        """
        match eq1:
            case 1:
                sol_in = (1 - shade) * solar_rad
                shortwave = sol_in - (sol_refl * sol_in)
            case 2:
                albedo = 0.05
                shortwave = (1 - albedo) * ((1 - shade) * solar_rad)
        return shortwave

    def _hflux_longwave(self, air_temp, water_temp, rel_hum, cl, vts):
        """
        Input:
        Note: must all be for the same time period and distance (ie same size)
        air_temp = an array of air temperature values (deg C)
        rel_hum = an array of relative humidity values (unitless)
        water_temp = an array of values for stream temperature (deg C)
        vts = view to sky coefficient (0-1)
        cl = cloud cover (0-1)
        """

        ### Stefan-Boltzman constant
        s_b = 5.67e-8

        e_s = np.vectorize(self._saturation_vp_air)(air_temp)
        e_a = (rel_hum / 100) * e_s
        e_atm = 1.72 * ((e_a / (air_temp + 273.2)) ** (1 / 7)) * (1 + 0.22 * cl**2)
        atm_rad = 0.96 * e_atm * vts * s_b * ((air_temp + 273.2) ** 4)

        back_rad = -0.96 * s_b * ((water_temp + 273.2) ** 4)
        land_rad = 0.96 * (1 - vts) * 0.96 * s_b * ((air_temp + 273.2) ** 4)
        longwave = atm_rad + back_rad + land_rad

        return longwave, atm_rad, back_rad, land_rad

    def _hflux_latent(
        self, water_temp, air_temp, rel_hum, wind_speed, shortwave, longwave, z, eq2
    ):
        """
        Input:
          Note: must all be for the same time period and distance (ie same size)
          shortwave = an array of values for solar radiation (W/m^2)
          longwave = an array of values for longwave radiation (W/m^2)
          rel_hum = an array of relative humidity values (unitless)
          water_temp = an array of values for stream temperature (deg C)
          wind_speed = an array of wind speed values (m/s)
          z = elevation of station where met data was obtained (m)
          eq2 = case1 uses the Penman Method to calculate latent heat flux
                case2 uses the Mass Transfer Method to calculate latent heat flux
                      *This switch is set in hflux_flux.m
        """
        c_air = 1004  # heat capacity of the air (J/kg deg C)
        rho_water = 1000  # Density of water (kg/m^3)
        rho_air = 1.2041  # Density of air at 20 deg C (kg/m^3)
        air_pressure = 101.3 - (0.0105 * z)  # air pressire
        psy_constant = (c_air * air_pressure) / (
            0.622 * 2450000
        )  # psychometric constant (kPa/deg C)
        # wind function coefficients
        b0 = 1.505e-8
        # 1/(m/s*kPa)
        b1 = 1.6e-8
        # 1/kPa

        e_s = np.vectorize(self._saturation_vp_air)(air_temp)
        e_a = (rel_hum / 100.0) * e_s
        l_e = 1000000 * (2.501 - (0.002361 * water_temp))

        match eq2:
            case 1:
                s = (4100 * e_s) / ((237 + air_temp) ** 2)
                r_a = 245 / ((0.54 * wind_speed) + 0.5)
                penman_evap = (
                    (
                        (s * (shortwave + longwave))
                        / (rho_water * l_e * (s + psy_constant))
                    )
                ) + (
                    (c_air * rho_air * psy_constant * (e_s - e_a))
                    / (rho_water * l_e * r_a * (s + psy_constant))
                )
                latent = (-rho_water) * l_e * penman_evap
            case 2:
                ews = np.vectorize(self._saturation_vp_water)(water_temp)
                fw = b0 + (b1 * wind_speed)
                mass_transfer_evap = fw * (ews - e_a)
                latent = -rho_water * l_e * mass_transfer_evap

        return latent

    def _hflux_sensible(
        self, water_temp, air_temp, rel_hum, wind_speed, z, latent, eq3
    ):
        """
        Input:
        Note: must all be for the same time period and distance (ie same size)
        air_temp = an array of air temperature values (deg C)
        rel_hum = an array of relative humidity values (unitless)
        water_temp = an array of values for stream temperature (deg C)
        wind_speed = an array of wind speed values (m/s)
        z = elevation of station where met data was obtained (m)
        latent = latent heat fulx (W/m^2)
        eq3 = case1 calculates the sensible heat transfer based on the Bowen ratio heat flux
              case2 calculates sensible heat transfer based on temperature differences,
                    Dingman, 1994
                    *This switch is set in hflux_flux.m
        """
        match eq3:
            case 1:
                ews = np.vectorize(self._sensible_saturated_vp)(water_temp)
                ewa = (rel_hum / 100) * ews
                air_pressure = 101.3 * (((293 - (0.0065 * z)) / 293) ** 5.256)
                br = 0.00061 * air_pressure * ((water_temp - air_temp) / (ews - ewa))
                sensible = br * latent
            case 2:
                c_air = 1004  # heat-capacity of the air (J/kg deg C)
                rho_air = 1.2041  # density of air at 20 degree C (kg/m^3)
                z_veg = 0.25  # height of vegetation around stream (m)
                z_met = 0.5  # height of wind speed measurements (m)

                zd = 0.7 * z_veg  # zero-plane displacement (m)
                z0 = 0.1 * z_veg  # roughness height (m)
                dh_dm = 1  # ratio of diffusivity of heat to diffusivity of momentum
                k = 0.4  # dimensionless constant

                kh = (
                    dh_dm
                    * c_air
                    * rho_air
                    * (k**2 / ((math.log((z_met - zd) / z0)) ** 2))
                )
                sensible = -kh * wind_speed * (water_temp - air_temp)

        return sensible

    def _hflux_bed(
        self, sed_type, water_temp, bed_temp, depth_of_measure, width_m, wp_m
    ):
        """
        Inputs:
            hflux_bed.m calculates the heat flux through the stream bed
            Inputs:
            sed_type = a single value or array of type 'cell' that describe the
                    sediment type as clay, sand, gravel, or cobbles
            water_temp: water tempature
            bed_temp: stream bed temprature measurements
            depth_of_meas: depth below the stream bed that temperature measurements
            were collected
            width_m: width of the stream (meters)
            WP_m: wetted perimeter of the stream bed (meters)
            Output:
            bed: the heat flux through the stream bed
        """
        k_sed = np.empty(len(sed_type))
        for i in range(len(sed_type)):
            match sed_type[i]:
                case 1:
                    k_sed[i] = 0.84  # (W/m*C)
                case 2:
                    k_sed[i] = 1.2  # (W/m*C)
                case 3:
                    k_sed[i] = 1.4  # (W/m*C)
                case 4:
                    k_sed[i] = 2.5  # (W/m*C)

        return (wp_m / width_m) * (
            -k_sed * ((water_temp - bed_temp) / depth_of_measure)
        )

    def _sensible_saturated_vp(self, water_temp_value):
        """
        Saturated vapor pressure using stream temp, used exclusively in sensible calculations
        """
        return 0.61275 * math.exp(
            (17.27 * water_temp_value) / (237.3 + water_temp_value)
        )

    def _saturation_vp_air(self, air_temp_value):
        """
        The saturation vapor pressure equation for air
        """
        return 0.611 * math.exp((17.27 * air_temp_value) / (237.2 + air_temp_value))

    def _saturation_vp_water(self, water_temp_value):
        """
        Saturated vapor pressure at the stream surface (water)
        """
        return 0.611 * math.exp((17.27 * water_temp_value) / (237.3 + water_temp_value))

    def heatflux_calculations(
        self,
        settings,
        solar_rad,
        air_temp,
        rel_hum,
        water_temp,
        wind_speed,
        z,
        sed_type,
        bed_temp,
        depth_of_meas,
        shade,
        vts,
        cl,
        sol_refl,
        wP_m,
        width_m,
    ):
        """
        Function that is called from hflux
        Input:
            settings = an array of values specifying solution methods
            solar_rad = an array of values for solar radiation (W/m^2)
            air_temp = an array of air temperature values (deg C)
            rel_hum = an array of relative humidity values (unitless)
            water_temp = an array of values for stream temperature (deg C)
            wind_speed = an array of wind speed values (m/s)
            z = elevation of station where met data was obtained (m)
            bed_temp=array of bed temperatures
            depth_of_meas=distance between stream temp measurement and stream bed
            temperature
            shade = amount of shade (0-1)
            cl = cloud cover (0-1)

        Output:
            net, shortwave, longwave, atm, back, land, latent, sensible, bed =
            heat fluxes to be employed by hflux
        """
        eq1 = settings[0][0]  # Shortwave radiation equation method
        ## 1 = Equation [3] in text, includes correction for reflection
        ## 2 = Equation [4] in text, includes albedo correction

        eq2 = settings[0][1]  # Latent Heat Flux Equation
        ## 1 = Equation [14] in text, Penman equation
        ## 2 = Equation [17] in text, mass transfer method

        eq3 = settings[0][2]  # Sensible Heat Equation
        ## 1 = Equation [20] in text, Bowen ratio method
        ## 2 = Equation [24] in text, from Dingman, 1994

        ### SHORTWAVE RADIATION
        shortwave = self._hflux_shortwave(solar_rad, shade, sol_refl, eq1)

        ### LONGWAVE RADIATION
        # Calculate longwave radiation
        # Westoff, et al. 2007/Boyd and Kasper, 2003
        longwave, atm_rad, back_rad, land_rad = self._hflux_longwave(
            air_temp, water_temp, rel_hum, cl, vts
        )

        ### LATENT HEAT
        # Calculate energy used for evaporation (latent heat) using the Penman
        # equation for open water
        latent = self._hflux_latent(
            water_temp, air_temp, rel_hum, wind_speed, shortwave, longwave, z, eq2
        )

        ### SENSIBLE HEAT
        # Calculate the sensible heat flux, which is the heat exchange between
        # the water and the air (driven by temperature differences)
        sensible = self._hflux_sensible(
            water_temp, air_temp, rel_hum, wind_speed, z, latent, eq3
        )

        ### STREAMBED CONDUCTION
        # Calculate the heat flux through the streambed
        bed = self._hflux_bed(
            sed_type, water_temp, bed_temp, depth_of_meas, width_m, wP_m
        )

        net = (shortwave + longwave + latent + sensible + bed) * 60
        return (
            net,
            shortwave,
            longwave,
            atm_rad,
            back_rad,
            land_rad,
            latent,
            sensible,
            bed,
        )

    def hflux_bed_sed(self, sed_type, dist_bed, dist_mod):
        """
        Inputs:
        sed_type = a single value or array of type 'cell' that describe the
                    sediment type as clay, sand, gravel, or cobbles
        dist_bed = distances in meters where the sediment type was observed
        dist_mod = interpolated distances in meters used in the model
        """
        sed_type_int = np.empty(len(sed_type))
        for index in range(len(sed_type)):
            match sed_type[index].lower():
                case "clay":
                    sed_type_int[index] = 1
                case "sand":
                    sed_type_int[index] = 2
                case "gravel":
                    sed_type_int[index] = 3
                case "cobbles":
                    sed_type_int[index] = 4
                case _:
                    print(
                        "Invalid sediment:",
                        sed_type[index],
                        "detected at index",
                        index,
                        ".",
                    )
                    sed_type[index] = -1  # error value, we can change later if we want

        return scipy.interpolate.interp1d(dist_bed, sed_type_int, "nearest")(dist_mod)

    def hflux_shortwave_refl(self, year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod):
        shortwave_reflection_calculations = ShortwaveReflectionCalculations()
        return shortwave_reflection_calculations.hflux_shortwave_refl(year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod)