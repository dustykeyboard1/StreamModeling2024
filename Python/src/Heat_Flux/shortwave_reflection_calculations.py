from datetime import datetime, timedelta
import numpy as np
import scipy
import math
import pytz

"""
Author: James Gallagher, Violet Shi, Michael Scoleri
File: shortwave_reflection_calculations.py
Date: 10-19-2023
Functionality: Compute Shortwave Reflection values
"""


class ShortwaveReflectionCalculations:
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
        time_frac = np.zeros(len(year))
        sol_zen = np.zeros(len(year))
        for i in range(len(year)):
            time_frac[i] = (hour[i] * (1 / 24)) + (minute[i] * (1 / 24) * (1 / 60))
            if self._daylight_savings_time(year[i], month[i], day[i], t_zone):
                time_frac[i] += 1 / 24
            sol_zen[i] = self._solar_position(
                time_frac[i],
                t_zone,
                year[i],
                month[i],
                day[i],
                hour[i],
                minute[i],
                lat,
                lon,
            )

        return scipy.interpolate.PchipInterpolator(
            time_met, self._fresnel_reflectivity(sol_zen)
        )(time_mod)

    def _fresnel_reflectivity(self, alpha_rad):
        """
        Computes fresnel's reflectivity

        Args:
            alpha_rad (ndarray): The alpha radiation reflecting off the stream.

        Returns:
            An ndarray of fresnel's reflectivity values for each radiation value.
        """
        n = 1.333
        ah = np.zeros(len(alpha_rad))
        for i in range(len(alpha_rad)):
            value = alpha_rad[i]
            if value < (math.pi / 2):
                beta_rad = math.asin(math.sin(value) / n)
                a_deg = ((math.tan(value - beta_rad)) ** 2) * (180 / math.pi)
                b_deg = ((math.tan(value + beta_rad)) ** 2) * (180 / math.pi)
                c_deg = ((math.sin(value - beta_rad)) ** 2) * (180 / math.pi)
                d_deg = ((math.sin(value + beta_rad)) ** 2) * (180 / math.pi)
                ah[i] = 0.5 * ((a_deg / b_deg) + (c_deg / d_deg))
            else:
                ah[i] = 1
        return ah

    def _solar_position(self, t_dst, t_zone, year, month, day, hour, minute, lat, lon):
        """
        Calculates the sun's position relative to the stream's position

        Args:
            t_dst (ndarray): A fraction representative of the hour and minute for a given measurement.
            t_zone (int): indicates the time zone correction factor (East = 5, Central = 6, Mountain = 7, Pacific = 8).
            year (ndarray): year during which measurements were taken.
            month (ndarray): month during which measurements were taken.
            day (ndarray): day during which measurements were taken.
            hour (ndarray): hour during which measurements were taken. Format is military time.
            min (ndarray): minute during which measurements were taken.
            lat (float): latitude. Positive for northern hemisphere, negative for southern.
            lon (float): longitude. Positive for eastern hemisphere, negative for western.

        Returns:
            An ndarray of the sun's position relative to the stream's.
        """

        t_gmt = t_dst + (t_zone / 24.0)
        a1 = round(year / 100)
        b1 = 2 - a1 + round(a1 / 4)
        t_jd = (
            round(365.25 * (year + 4716))
            + round(30.6001 * (month + 1))
            + day
            + b1
            - 1524.5
        )
        t_jdc = (t_jd + t_gmt - 2451545) / 36525

        # Solar position relative to Earth
        s = 21.448 - t_jdc * (46.815 + t_jdc * (0.00059 - (t_jdc * 0.001813)))
        o_ob_mean = 23 + ((26 + (s / 60)) / 60)
        o_ob = o_ob_mean + 0.00256 * math.cos(125.04 - 1934.136 * t_jdc * math.pi / 180)
        e_c = 0.016708634 - t_jdc * (0.000042037 + 0.0000001267 * t_jdc)
        o_ls_mean = (280.46646 + t_jdc * (36000.76983 + 0.0003032 * t_jdc)) % 360
        o_as_mean = 357.52911 + t_jdc * (35999.05029 - 0.0001537 * t_jdc)
        a = o_as_mean * math.pi / 180
        b = math.sin(a)
        c = math.sin(b * 2)
        d = math.sin(c * 3)
        o_cs = (
            b * (1.914602 - t_jdc * (0.004817 + 0.000014 * t_jdc))
            + c * (0.019993 - 0.000101 * t_jdc)
            + d * 0.000289
        )
        o_ls = o_ls_mean + o_cs
        o_al = (
            o_ls
            - 0.00569
            - (0.00478 * math.sin((125.04 - 1934.136 * t_jdc) * math.pi / 180))
        )
        solar_dec = (
            math.asin(math.sin(o_ob * math.pi / 180) * math.sin(o_al * math.pi / 180))
            * 180
            / math.pi
        )
        o_ta = o_as_mean + o_cs

        ## A -> e; B -> f; C -> g; D -> h; E -> i; F -> j

        e = (math.tan(o_ob * math.pi / 360)) ** 2
        f = math.sin(2 * o_ls_mean * math.pi / 180)
        g = math.sin(o_as_mean * math.pi / 180)
        h = math.cos(2 * o_ls_mean * math.pi / 180)
        i = math.sin(4 * o_ls_mean * math.pi / 180)
        j = math.sin(2 * o_as_mean * math.pi / 180)
        e_t = (
            4
            * (
                e * f
                - 2 * e_c * g
                + 4 * e_c * e * g * h
                - 0.5 * (e**2) * i
                - (4 / 3) * (e_c**2) * j
            )
            * (180 / math.pi)
        )
        lstm = 15 * round(lon / 15)
        t_s = ((hour * 60) + minute) + 4 * (lstm - lon) + e_t
        o_ha = t_s / 4 - 180

        # Solar position relative to stream position
        m = math.sin(lat * (math.pi / 180)) * math.sin(
            solar_dec * (math.pi / 180)
        ) + math.cos(lat * (math.pi / 180)) * math.cos(
            solar_dec * (math.pi / 180)
        ) * math.cos(
            o_ha * (math.pi / 180)
        )
        return math.acos(m)

    def _daylight_savings_time(self, year, month, day, t_zone):
        """
        Determines if dst is active for the current year, month, day, time zone parameters
        CITE: https://thispointer.com/check-if-date-is-daylight-saving-in-python/
        DESC: Used to determine if dst is active

        Args:
            year (ndarray): year during which measurements were taken.
            month (ndarray): month during which measurements were taken.
            day (ndarray): day during which measurements were taken.
            t_zone (int): indicates the time zone correction factor (East = 5, Central = 6, Mountain = 7, Pacific = 8).

        Returns:
            True if DST is active, and False otherwise
        """
        timezone = pytz.timezone(self._match_timezones(t_zone))
        (
            year,
            month,
            day,
        ) = (
            int(year.item()),
            int(month.item()),
            int(day.item()),
        )  # converting numpy vals to python ints
        date = timezone.localize(datetime(year, month, day))
        return date.dst() != timedelta(0)

    def _match_timezones(self, t_zone):
        """
        Matches the time zones from a numerical input into a pytz input so we can determine dst

        Args:
            t_zone (int): indicates the time zone correction factor (East = 5, Central = 6, Mountain = 7, Pacific = 8).

        Returns:
            A string corresponding to the time zone integer
        """
        match t_zone:
            case 5:
                return "US/Eastern"
            case 6:
                return "US/Central"
            case 7:
                return "US/Mountain"
            case 8:
                return "US/Pacific"
            case _:
                return None
