import pandas as pd
import numpy as np
import math
import datetime 

# % Input:
# %   Note: must all be for the same time period and distance (ie same size)
# %   year = year during which measurements were taken (ie. 2012)
# %   month = month during which measurements were taken (ie. 6)
# %   day = day during which measurements were taken (ie. 18)
# %   hour = hour during which measurements were taken (ie. 22) (military
# %      time)
# %   min = minute during which measurements were taken (ie. 5)
# %   time_met = times at which meteorological data are known, in minutes (met
# %       data includes air temp, rel humidity, wind speed, and solar rad)
# %   lat = latitude (positive for northern hemisphere, negative for southern)
# %   lon = longitude (positive for eastern hemisphere, negative for western)
# %   t_zone = indicates the time zone correction factor (East = 5, Central = 6,
# %       Mountain = 7, Pacific = 8)
# %   time_mod = model times at which temperatures will be computed for heat budget,
# %   in minutes
# %
# % Output:
# %  [sol_refl] = portion of solar radiation reflected off the surface of the
# %  stream

def hflux_shortwave_relf(year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod):
    time_frac = np.zeros(len(year))
    sol_zen = np.zeros(len(year))
    for i in range(len(year)):
        time_frac[i] = (hour[i] * (1/24)) + (minute[i] * (1/24) * (1/60))
        if daylight_saving_time(year[i], month[i], day[i]):
            time_frac[i] += (1/24)
        sol_zen[i] = solar_position(time_frac[i], t_zone, year[i], month[i], day[i], hour[i], minute[i], lat, lon)
    
    return fresnel_reflectivity(sol_zen)

### Calculate freznel's reflectivity
def fresnel_reflectivity(alpha_rad):
    n = 1.333
    ah = np.zeros(len(alpha_rad))
    for i in range(len(alpha_rad)):
        value = alpha_rad[i]
        if (value < (math.pi / 2)):
            beta_rad = math.asin(math.sin(value) / n)
            a_deg = ((math.tan(value - beta_rad)) ** 2) * (180 / math.pi)
            b_deg = ((math.tan(value + beta_rad)) ** 2) * (180 / math.pi)
            c_deg = ((math.sin(value - beta_rad)) ** 2) * (180 / math.pi)
            d_deg = ((math.sin(value + beta_rad)) ** 2) * (180 / math.pi)
            print(a_deg, b_deg, c_deg, d_deg)
            ah[i] = .5 * ((a_deg / b_deg) + (c_deg / d_deg))
        else:
            ah[i] = 1
    return ah
    
### Calculates the solar position relative to the stream's position
### Returns sol_zen, which is something, but we need it for later fs fs
def solar_position(t_dst, t_zone, year, month, day, hour, minute, lat, lon):
    t_gmt = t_dst + (t_zone / 24.0)
    a1 = round(year / 100)
    b1 = 2 - a1 + round(a1 / 4)
    t_jd = round(365.25 * (year + 4716)) + round(30.6001 * (month + 1)) + day + b1 - 1524.5
    t_jdc = (t_jd + t_gmt - 2451545) / 36525
    
    # Solar position relative to Earth
    s = 21.448 - t_jdc * (46.815 + t_jdc * (.00059 - (t_jdc * .001813)))
    o_ob_mean = 23 + ((26 + (s / 60)) / 60)
    o_ob = o_ob_mean + .00256 * math.cos(125.04 - 1934.136 * t_jdc * math.pi / 180)
    e_c = .016708634 - t_jdc * (.000042037 + .0000001267 * t_jdc)
    o_ls_mean = (280.46646 + t_jdc * (36000.76983 + .0003032 * t_jdc)) % 360
    o_as_mean = 357.52911 + t_jdc * (35999.05029 - .0001537 * t_jdc)
    a = (o_as_mean * math.pi/180)
    b = math.sin(a)
    c = math.sin(b * 2)
    d = math.sin(c * 3)
    o_cs = b * (1.914602 - t_jdc * (.004817 + .000014 * t_jdc)) + c * (.019993 - .000101 * t_jdc) + d * .000289
    o_ls = o_ls_mean + o_cs
    o_al = o_ls - .00569 - (.00478 * math.sin((125.04 - 1934.136 * t_jdc) * math.pi / 180))
    solar_dec = math.asin(math.sin(o_ob * math.pi / 180) * math.sin(o_al * math.pi / 180)) * 180 / math.pi  ##QUESTION
    o_ta = o_as_mean + o_cs

    ## A -> e; B -> f; C -> g; D -> h; E -> i; F -> j 

    e = (math.tan(o_ob * math.pi / 360)) ** 2
    f = math.sin(2 * o_ls_mean * math.pi / 180)
    g = math.sin(o_as_mean * math.pi / 180)
    h = math.cos(2 * o_ls_mean * math.pi / 180)
    i = math.sin(4 * o_ls_mean * math.pi / 180)
    j = math.sin(2 * o_as_mean * math.pi / 180)
    e_t = 4 * (e * f -2 * e_c * g + 4 * e_c * e * g * h -.5 * (e **2) * i - (4/3) * (e_c ** 2) * j) * (180 / math.pi)
    lstm = 15 * round(lon / 15)
    t_s = ((hour * 60) + minute) + 4 * (lstm - lon) + e_t
    o_ha = t_s / 4 - 180
    
    # Solar position relative to stream position
    m = math.sin(lat * (math.pi / 180)) * math.sin(solar_dec * (math.pi / 180)) + math.cos(lat * (math.pi / 180)) * math.cos(solar_dec * (math.pi / 180)) * math.cos(o_ha * (math.pi / 180))
    return math.acos(m)

### Calculates whether we need to apply a daylight savings time for the reflection calcs
def daylight_saving_time(year, month, day):
        # daylight savings correction
        if year < 2007:   
            if month > 4 and month < 10: #if the months are between April and October, true
                return True
            elif month < 4 or month > 10: # if the months are January, Fenruary, March, November, or December, False
                return False
            if month == 4: # if the month is April
                ### Want the first Sunday 
                return sunday_dst_calculator(year, month, day, 0)
        
            elif month == 10: # if the month is October
                ### Want the last Sunday
                ### The "not" is here to account for the function returning "True" if we are before or
                ### on the last sunday in october. Since we want to check whether we are after that sunday
                ### we can negate it to get a correct value
                return not sunday_dst_calculator(year, month, day, -1)

        else:
            if month > 3 and month < 11: #if the months are between April and October, true
                return True
            elif month < 3 or month > 11: # if the months are January, Fenruary, or December, False
                return False
           
            elif month == 3: #if the month is March
                return sunday_dst_calculator(year, month, day, 1)
           
            elif month == 11: # if the month is November
                ### The "not" is here to account for the function returning "True" if we are before or
                ### on the first sunday in novermber. Since we want to check whether we are after that sunday
                ### we can negate it to get a correct value
                return not sunday_dst_calculator(year, month, day, 0)


### Returns whether your day is after the sunday cutoff for DST
def sunday_dst_calculator(year, month, day, desired_sunday):
    sunday = []
    for i in range(1, days_in_month(month) + 1): #days in the month from [1, 30/31]
        day_of_week = datetime.date(year, month, i).weekday()
        if day_of_week == 6: # if it is a sunday
            sunday.append(i)
    
    if day >= sunday[desired_sunday]:
        return True
    return False

### Returns the days in the relevant month
def days_in_month(month):
    match month:
        case 4 | 11:
            return 30 # April and Novemeber have 30 days
        case 3 | 10:
            return 31 # March and Ocotber have 31 days