import numpy as np
import pandas as pd
import math
import scipy
from .shortwave_reflection_calculations import ShortwaveReflectionCalculations

"""
Author: James Gallagher, Violet Shi, Michael Scoleri
File: heatflux_calculations.py
Date: 10-19-2023
Functionality: Calculates the total heat entering and leaving a stream
for a width of stream section over time
"""


class HeatFluxCalculations:
    def _hflux_shortwave(self, solar_rad, shade, sol_refl, eq1):
        """
        Computes shortwave radiation impacting the stream's surface.

        Args:
            solar_rad (ndarray): total incoming solar radiation data at each time_met.
            shade (float): values for shading (0 to 1, with 0 being min shading and 1 being max shading).
            sol_refl (ndarray): portion of solar radiation that is reflected off the surface of the stream.
            eq1 (int): case1 uses the Ouellet, et al. 2012 and Boyd and Kasper, 2003 methods;
                       case2 uses the Magnusson, et al. 2012 nethod.

        Returns:
            shortwave (ndarray): an array shortwave radiation with units (W/m^2).
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
        Computes longwave radiation impacting the stream's surface.

        Args:
            air_temp (ndarray): air temperature values (deg C)
            rel_hum (ndarray): relative humidity values (unitless)
            water_temp (ndarray): values for stream temperature (deg C)
            vts (ndarry): view to sky coefficient (0-1)
            cl (ndarray): cloud cover (0-1)

        Returns:
            longwave (ndarray): an array longwave radiation with units (W/m^2).
            atm_rad (ndarray): atmospheric longwave radiation
            back_rad (ndarray): back radiation from the stream
            land_rad (ndarray): radiation from the landcover
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
        Computes the latent heat flux for the stream

        Args:
            shortwave (ndarray): an array of values for solar radiation (W/m^2)
            longwave (ndarray): an array of values for longwave radiation (W/m^2)
            rel_hum (ndarray): an array of relative humidity values (unitless)
            water_temp (ndarray): an array of values for stream temperature (deg C)
            wind_speed (ndarray): an array of wind speed values (m/s)
            z (int): elevation of station where met data was obtained (m)
            eq2 (int): case1 uses the Penman Method to calculate latent heat flux
                       case2 uses the Mass Transfer Method to calculate latent heat flux

        Returns:
            latent (ndarray): latent heat flux values for each position along the stream
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
        Computes the sensible heat flux for the stream

        Args:
            air_temp (ndarray): an array of air temperature values (deg C)

            rel_hum (ndarray): an array of relative humidity values (unitless)
            water_temp (ndarray): an array of values for stream temperature (deg C)
            wind_speed (ndarray): an array of wind speed values (m/s)
            z (int): elevation of station where met data was obtained (m)
            latent (ndarray): latent heat fulx (W/m^2)
            eq3 (int): case1 calculates the sensible heat transfer based on the Bowen ratio heat flux
                       case2 calculates sensible heat transfer based on temperature differences, Dingman, 1994
        Returns:
            sensible (ndarray): sensible heat flux values for each position along the stream
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
        Calculates the heat flux through the stream bed

        Args:
            sed_type (ndarray): sediment type of the stream. Accepted values are: clay, sand, gravel, or cobbles.
            water_temp (ndarray): water tempature
            bed_temp (ndarray): stream bed temprature measurements
            depth_of_meas (ndarray): depth below the stream bed that temperature measurements
            were collected
            width_m (ndarray): width of the stream (meters)
            WP_m (ndarray): wetted perimeter of the stream bed (meters)

        Returns:
            bed (ndarray): the heat flux through the stream bed at each location
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
        Saturated vapor pressure using the stream's water temperature.
        Used exclusively in sensible calculations.

        Args:
            water_temp_value (float): A singluar value of water_temp

        Returns:
            The Saturation vapor pressure of that water_temp value
        """
        return 0.61275 * math.exp(
            (17.27 * water_temp_value) / (237.3 + water_temp_value)
        )

    def _saturation_vp_air(self, air_temp_value):
        """
        Saturated vapor pressure using the stream's air temperature.

        Args:
            water_temp_value (float): A singluar value of air_temp

        Returns:
            The Saturation vapor pressure of that air temperature value
        """
        return 0.611 * math.exp((17.27 * air_temp_value) / (237.2 + air_temp_value))

    def _saturation_vp_water(self, water_temp_value):
        """
        Saturated vapor pressure using the stream's water temperature.

        Args:
            water_temp_value (float): A singluar value of water_temp

        Returns:
            The Saturation vapor pressure of that water_temp value
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
        Computes the total heat entering and leaving a stream

        Args:
            settings (ndarray): an array of values specifying solution methods
            solar_rad (ndarray): an array of values for solar radiation (W/m^2)
            air_temp (ndarray): an array of air temperature values (deg C)
            rel_hum (ndarray): an array of relative humidity values (unitless)
            water_temp (ndarray): an array of values for stream temperature (deg C)
            wind_speed (ndarray): an array of wind speed values (m/s)
            z (int)= elevation of station where met data was obtained (m)
            sed_type (ndarray): sediment type of the stream. Accepted values are: clay, sand, gravel, or cobbles.
            bed_temp (ndarray): array of bed temperatures
            depth_of_meas (ndarray): distance between stream temp measurement and stream bed
            temperature
            shade (ndarray): amount of shade (0-1)
            vts (ndarry): view to sky coefficient (0-1)
            cl (ndarray): cloud cover (0-1)
            sol_refl (ndarray): portion of solar radiation that is reflected off the surface of the stream.
            WP_m (ndarray): wetted perimeter of the stream bed (meters)
            width_m (ndarray): width of the stream (meters)

        Returns:
            net (ndarry): summed heat fluxe values from shortwave, longwave, latent, sensible, and bed
            shortwave (ndarry): shortwave radiation values
            longwave (ndarry): longwave radiation values
            atm (ndarray): atmospheric longwave radiation
            back (ndarray): back radiation from the stream
            land (ndarray): radiation from the landcover
            latent (ndarray): latent heat flux values for each position along the stream
            sensible (ndarray): sensible heat flux values for each position along the stream
            bed (ndarray): the heat flux through the stream bed at each location
        """
        eq1 = settings[
            "shortwave radiation method"
        ]  # Shortwave radiation equation method
        ## 1 = Equation [3] in text, includes correction for reflection
        ## 2 = Equation [4] in text, includes albedo correction

        eq2 = settings["latent heat flux equation"]  # Latent Heat Flux Equation
        ## 1 = Equation [14] in text, Penman equation
        ## 2 = Equation [17] in text, mass transfer method

        eq3 = settings["sensible heat equation"]  # Sensible Heat Equation
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
        Interpolates bed sediment input to the entire length of the stream

        Args:
            sed_type (ndarray): sediment type of the stream. Accepted values are: clay, sand, gravel, or cobbles.
            dist_bed (ndarray): distances in meters where the sediment type was observed.
            dist_mod (ndarray): interpolated distances in meters used in the model.

        Returns:
            An array of interpolated values for the sediment type

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

    def hflux_shortwave_refl(
        self, year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod
    ):
        """
        Calculate the portion of solar radiation reflected off the surface of the stream

        Args:
            year (ndarray): year during which measurements were taken.
            month (ndarray): month during which measurements were taken.
            day (ndarray): day during which measurements were taken.
            hour (ndarray): hour during which measurements were taken. Format is military time.
            min (ndarray): minute during which measurements were taken.
            time_met (ndarray): times at which meteorological data are known, in minutes. met
                data includes air temp, rel humidity, wind speed, and solar rad.
            lat (float): latitude. Positive for northern hemisphere, negative for southern.
            lon (float): longitude. Positive for eastern hemisphere, negative for western.
            t_zone (int)= indicates the time zone correction factor (East = 5, Central = 6, Mountain = 7, Pacific = 8).
            time_mod (ndarray): model times at which temperatures will be computed for heat budget, in minutes.

        Returns:
            sol_refl (ndarray): an array of solar radiation
        """
        shortwave_reflection_calculations = ShortwaveReflectionCalculations()
        return shortwave_reflection_calculations.hflux_shortwave_refl(
            year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod
        )
