import numpy as np
import os
import sys
import math
from datetime import datetime

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities import Input_reader
from src.Heat_Flux.hflux_bed_sed import hflux_bed_sed
from src.Heat_Flux.hflux_shortwave_refl import hflux_shortwave_relf
from src.Heat_Flux.hflux_flux import hflux_flux
from src.Utilities.interpolation import interpolation

def hflux():
    # read from excel sheet
    filename = os.path.join(os.getcwd(), 'Data', 'example_data.xlsx')
    input_data = Input_reader.readFromFile(filename)

    print('Assigning variable names...')

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
    # print(sed_type)
    
    sol_refl = hflux_shortwave_relf(year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod)

    print('...done!')
    print('Determining time steps and nodes...')

    timesteps = len(time_mod)
    dt = max(time_mod) / (len(time_mod) - 1)
    
    print('...done!')

    print('Interpolating longitudinal data in space...')

    ### Interpolate Data
    t_l_m = interpolation(dist_T_L, t_L, dist_mod)
    temp_t0_m = interpolation(dist, temp_t0, dist_mod)
    ### Need to transpose discharge_m to make sure it has the same shape
    ### As discharge_m in Matlab. I do not know why, but this works
    ### And the values are correct

    discharge_m = interpolation(dist_dis, discharge, dist_mod).transpose()

    width_m = interpolation(dist_stdim, width, dist_mod)
    depth_m = interpolation(dist_stdim, depth, dist_mod)
    depth_of_meas_m = interpolation(dist_bed, depth_of_meas, dist_mod)
    shade_m = interpolation(dist_shade, shade, dist_mod)
    vts_m = interpolation(dist_shade, vts, dist_mod)

    ### This works, would be cool if there was a more elegant solution
    ### I could not find one, however
    
    # checked!
    bed_temp_m = [0] * len(time_bed)
    for i in range(len(time_bed)):
        bed_temp_m[i] = interpolation(dist_bed, bed_temp[i], dist_mod)
    bed_temp_m = np.array(bed_temp_m).transpose()

    print('...done!')

    print('Interpolating temporal data in time...')
    ### Interpolate all data given through time so that there are 
    ### Values at every step
    # checked!
    discharge_m = interpolation(time_dis, discharge_m, time_mod)

    ### Calculate width-depth-discharge relationship
    r = len(dist_mod)
    # checked!
    
    # theta = np.empty(r)
    theta = np.arctan((.5 * width_m) / depth_m)
    
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    # checked!
    dim_q = interpolation(dist_stdim, discharge_stdim, dist_mod)
    n_s = ((.25 ** (2/3)) * (2 ** (5/3)) * 
           (cos_theta ** (2/3)) * 
           (tan_theta ** (5/3)) * (depth_m ** (8/3))) / (2 * dim_q)

    # transpose discharge_m for calculation purpose
    # depends on how the calculations go later on in the steps
    # can possibly stay in the transpose position
    # for now, discharge_m is transposed back on line 158
    discharge_m = discharge_m.transpose()

    # checked!
    depth_m = (((2 * n_s * discharge_m)/
               ((0.25**(2/3))*(2**(5/3)) * 
                (cos_theta**(2/3)) *
                (tan_theta**(5/3))))**(3/8))

    # checked! 
    width_m = (2 * 
               tan_theta * 
               (((2 * n_s * discharge_m) / 
                ((0.25**(2/3)) * (2**(5/3)) * 
                (cos_theta**(2/3)) *
                (tan_theta**(5/3))))**(3/8)))
    
    # checked!
    area_m = 0.5 * depth_m * width_m
    area_m = area_m.transpose()
    # checked!
    wp_m = (2 * (depth_m / cos_theta)).transpose()
    width_m = width_m.transpose()
    depth_m = depth_m.transpose()

    # tranpose discharge_m back to its original shape
    discharge_m = discharge_m.transpose()

    # all checked!
    solar_rad_dt = interpolation(time_met, solar_rad_in, time_mod)
    air_temp_dt = interpolation(time_met, air_temp_in, time_mod)
    rel_hum_dt = interpolation(time_met, rel_hum_in, time_mod)
    wind_speed_dt = interpolation(time_met, wind_speed_in, time_mod)
    c_dt = interpolation(time_cloud, c_in, time_mod)
    temp_x0_dt = interpolation(time_temp, temp_x0, time_mod, 'pchip')

    # checked!
    bed_temp_dt = [0] * r
    for i in range(r):
        bed_temp_dt[i] = interpolation(time_bed, bed_temp_m[i], time_mod)
    bed_temp_dt = np.array(bed_temp_dt)

    # checked!
    solar_rad_mat = np.array([solar_rad_dt] * r)
    air_temp_mat= np.array([air_temp_dt] * r)
    rel_hum_mat = np.array([rel_hum_dt] * r)
    wind_speed_mat = np.array([wind_speed_dt] * r)
    cl = np.array([c_dt] * r)

    print('...done!')

    ###############################################################
    # STEP 1: compute volumes of each reservoir (node) in the model
    # checked!
    print('Computing volumes of nodes, discharge rates and groundwater inflow rates...')
    volume = np.empty((r, timesteps))
    volume[0] = (dist_mod[1] - dist_mod[0]) * area_m[0]
    volume[1: r-1] = (area_m[1:r-1].transpose() * 
                      (((dist_mod[2:] + dist_mod[1:r-1]) / 2) - 
                       ((dist_mod[1:r-1] + dist_mod[:r-2]) / 2))).transpose()
    volume[r-1] = area_m[r-1] * (dist_mod[r-1]-dist_mod[r-2])

    ###############################################################
    # STEP 2: compute discharge rates at reservoir edges using linear
    # interpolation (n-values are upstream of node n values)
    # checked!
    q_half = np.empty((r+1, timesteps))
    q_half[0] = (2 * discharge_m[0] - discharge_m[1] + discharge_m[0]) / 2
    q_half[1:r] = (discharge_m[1:r]+discharge_m[0:r-1]) / 2
    q_half[r] = (2 * discharge_m[r-1] - discharge_m[r-2] + discharge_m[r-1]) / 2

    ###############################################################
    # STEP 3: compute lateral groundwater discharge rates to each node based on
    # longtudinal changes in streamflow
    # checked!
    q_l = q_half[1:r + 1] - q_half[:r]

    ###############################################################
    # STEP 4: unit conversions so all discharge rates are in m3/min
    # note that all inputs are x are in m, T are in deg C, and Q or Q_L are in m3/s
    # checked!
    q_half_min = q_half * 60
    q_l_min = q_l * 60

    print('...done!')

    ###############################################################
    # STEP 5: Calculate coefficients of the tridiagonal matrix (a, b, c)
    # and set coefficients at the boundaries. Use a, b and c to create the A
    # matrix.  Note that a, b and c are constant in time as long as Q,
    # volume, and Q_L are constant with time.
    # checked!
    double_volume = 2 * volume[:, 0]
    quad_volume = 4 * volume[:, 0]
    a = np.empty((r, timesteps))
    a[:, :timesteps-1] = ((-dt * q_half_min[:r, 1:]).transpose() / quad_volume).transpose()
    a[:, timesteps-1] = (-dt * q_half_min[:r, timesteps-1]) / quad_volume
    
    # checked!
    b = np.empty((r, timesteps))
    o1 = (dt * q_half_min[:r, 1:]).transpose() / quad_volume
    p1 = (dt * q_half_min[1:, 1:]).transpose() / quad_volume
    q1 = (dt * q_l_min[:,:timesteps-1]).transpose() / double_volume
    o2 = (dt * q_half_min[:r, timesteps-1]).transpose() / quad_volume
    p2 = (dt * q_half_min[1:, timesteps-1]).transpose() / quad_volume
    q2 = (dt * q_l_min[:r, timesteps-1]).transpose() / double_volume
    b[:, :timesteps-1] = (1 + o1 - p1 + q1).transpose()
    b[:, timesteps-1] = (1 + o2 - p2 + q2).transpose()
    
    # checked!
    c = np.empty((r, timesteps))
    c[:, :timesteps-1] = ((dt * q_half_min[1:, 1:]).transpose() / quad_volume).transpose()
    c[:, timesteps-1] = (dt * q_half_min[1:, timesteps-1]) / quad_volume

    # all checked!
    a_c = ((-dt * q_half_min[:r]).transpose() / quad_volume).transpose()
    o_c = (dt * q_half_min[:r]).transpose() / quad_volume
    p_c = (dt * q_half_min[1:,:]).transpose() / quad_volume
    q_c = (dt * q_l_min).transpose() / double_volume
    b_c = (1 + o_c - p_c + q_c).transpose()
    c_c = ((dt * q_half_min[1:,:]).transpose() / quad_volume).transpose()

    ###############################################################
    # STEP 6: Calculate right hand side (d).
    # The values for d are temperature-dependent, so they change each time step.
    # Once d is computed, use that d value and the
    # matrix A to solve for the temperature for each time step.
    print('Computing d-values, heat fluxes and solving for stream temperatures...')
    
    d = np.empty((r, timesteps))
    t = np.empty((r, timesteps))
    heat_flux = np.empty((r, timesteps))
    shortwave = np.empty((r, timesteps))
    longwave = np.empty((r, timesteps))
    atm = np.empty((r, timesteps))
    back = np.empty((r, timesteps))
    land = np.empty((r, timesteps))
    latent = np.empty((r, timesteps))
    sensible = np.empty((r, timesteps))
    bed = np.empty((r, timesteps))

    # print(sed)
    heat_flux[:, 0], shortwave[:, 0], longwave[:, 0], atm[:, 0], back[:, 0], land[:, 0], latent[:, 0], sensible[:, 0], bed[:, 0] = hflux_flux(input_data["settings"], solar_rad_mat[:, 0],
                                             air_temp_mat[:, 0], rel_hum_mat[:, 0], temp_t0_m,
                                             wind_speed_mat[:, 0], z, sed, bed_temp_dt[:, 0],
                                             depth_of_meas_m, shade_m, vts_m,
                                             cl[:, 0], sol_refl[0], wp_m[:r, 0], width_m[:, 0])
    rho_water = 1000
    c_water = 4182

    print("Calculating...")

    g = 1 + a_c + c_c - q_c.transpose()

    # Could be done better potentially, leave it for alpha for now
    k = np.empty((r, timesteps))
    for i in range(r):
        for j in range(timesteps):
            if q_l_min[i, j] < 0:
                k[i, j] = 0
            else:
                k[i, j] = (dt * q_l_min[i,j]) / (volume[i, 0])
    
    print(k)
hflux()
