import numpy as np
import os
import sys
import math

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities import Input_reader
from src.Heat_Flux.hflux_bed_sed import hflux_bed_sed
from src.Heat_Flux.hflux_shortwave_refl import hflux_shortwave_relf
from src.Utilities.interpolation import interpolation

def hflux():
    # read from excel sheet
    filename = os.getcwd() + "/Python/Data/example_data.xlsx"
    input_data = Input_reader.readFromFile(filename)

    method = input_data["settings"][0][0]
    unattend = input_data["settings"][0][4]

    # initialize variables
    time_mod = input_data["time_mod"][0]
    dist_mod = input_data["dist_mod"][0]

    time_temp = input_data["temp_x0_data"][0]
    temp_x0 = input_data["temp_x0_data"][1]

    dist = input_data["temp_t0_data"][0]
    temp_t0 = input_data["temp_t0_data"][1]

    dist_stdim = input_data["dim_data"][0]
    # area = input_data["dim_data"][1]  p.s. commented out in the original Matlab file
    width = input_data["dim_data"][2]
    depth = input_data["dim_data"][3]
    discharge_stdim = input_data["dim_data"][4]

    dist_dis = input_data["dis_data"][0]
    # discharge is multiple columns
    # I changed this from the old method since this allows for any number
    # Of columns after the first one
    discharge = input_data["dis_data"][1:]
    time_dis = input_data["time_dis"][0]

    dist_T_L = input_data["T_L_data"][0]
    t_L = input_data["T_L_data"][1]

    year = input_data["met_data"][0]
    month = input_data["met_data"][1]
    day = input_data["met_data"][2]
    hour = input_data["met_data"][3]
    minute = input_data["met_data"][4]
    time_met = input_data["met_data"][5]
    solar_rad_in = input_data["met_data"][6]
    air_temp_in = input_data["met_data"][7]
    rel_hum_in = input_data["met_data"][8]
    wind_speed_in = input_data["met_data"][9]

    dist_bed = input_data["bed_data1"][0]
    depth_of_meas = input_data["bed_data1"][1]

    time_bed = np.array([input_data["bed_data2"][0, 0], input_data["bed_data2"][1, 0]])
    ### Same initialization, just allows for an infinite number of potential rows / columns
    bed_temp = input_data["bed_data2"][0:, 1:]

    sed_type = input_data["sed_type"][0]

    dist_shade = input_data["shade_data"][0]
    shade = input_data["shade_data"][1]
    vts = input_data["shade_data"][2]

    time_cloud = input_data["cloud_data"][0]
    c_in = input_data["cloud_data"][1]

    lat = input_data["site_info"][0, 0]
    lon = input_data["site_info"][0, 1]
    t_zone = input_data["site_info"][0, 2]
    z = input_data["site_info"][0, 3]

    sed = hflux_bed_sed(sed_type, dist_bed, dist_mod)
    sol_relf = hflux_shortwave_relf(year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod)

    timesteps = len(time_mod)
    dt = max(time_mod) / (len(time_mod) - 1)
    
    ### Interpolate Data
    t_l_m = interpolation(dist_T_L, t_L, dist_mod)
    temp_t0_m = interpolation(dist, temp_t0, dist_mod)
    ### Need to transpose discharge_m to make sure it has the same shape
    ### As discharge_m in Matlab. I do not know why, but this works
    ### And the values are correct
    discharge_m = interpolation(dist_dis, discharge, dist_mod).transpose()

    # print(discharge_m)
    width_m = interpolation(dist_stdim, width, dist_mod)
    depth_m = interpolation(dist_stdim, depth, dist_mod)
    depth_of_meas_m = interpolation(dist_bed, depth_of_meas, dist_mod)
    shade_m = interpolation(dist_shade, shade, dist_mod)
    vts_m = interpolation(dist_shade, vts, dist_mod)

    ### This works, would be cool if there was a more elegant solution
    ### I could not find one, however
    bed_temp_m = [0] * len(time_bed)
    for i in range(len(time_bed)):
        bed_temp_m[i] = interpolation(dist_bed, bed_temp[i], dist_mod)
    bed_temp_m = np.array(bed_temp_m).transpose()
    # print(bed_temp_m, bed_temp_m.shape)

    ### Interpolate all data given through time so that there are 
    ### Values at every step

    discharge_m = interpolation(time_dis, discharge_m, time_mod).transpose()
    ### Since we already transposed discharge_m, we do not have to do it 
    ### Again. Strange quicks converting to numpy, but the bottom line,
    ### (And I HAVE NOT) checked this, is that by this point,
    ### Every variable we create should have exactly the same dimensions
    ### As the variables in MatLab that we created these from

    ### Calculate width-depth-discharge relationship
    r = len(dist_mod)
    theta = np.zeros(r)
    for i in range(r):
        theta[i] = math.atan((.5 * width_m[i]) / depth_m[i])
    
    dim_q = interpolation(dist_stdim, discharge_stdim, dist_mod)
    n_s = np.zeros(r)
    for i in range(r):
        n_s[i] = ((.25 ** (2/3)) * (2 ** (5/3)) * (math.cos(theta[i]) ** (2/3)) * (math.tan(theta[i]) ** (5/3)) * (depth_m[i] ** (8/3))) / (2 * dim_q[i])

    

hflux()
