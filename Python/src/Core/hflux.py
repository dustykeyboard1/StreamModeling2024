"""
Author: Violet Shi, James Gallagher, Michael Scoleri
Date: 10-02-23
File: hflux.py
Functionality: Implementation of hflux.m
"""
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities import Input_reader
from src.Heat_Flux.heatflux_calculations import HeatFluxCalculations
from src.Utilities.interpolation import interpolation
from src.Plotting.plotting_class import Plotting as plc


def hflux(input_data):
    hflux_calculations = HeatFluxCalculations()
    method = input_data["settings"][0][0]
    unattend = bool(input_data["settings"][0][4])

    if not unattend:
        print("Assigning variable names...")

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

    sed = hflux_calculations.hflux_bed_sed(sed_type, dist_bed, dist_mod)

    sol_refl = hflux_calculations.hflux_shortwave_refl(
        year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod
    )

    if not unattend:
        print("...done!")
        print()
        print("Determining time steps and nodes...")

    timesteps = len(time_mod)
    dt = max(time_mod) / (len(time_mod) - 1)

    if not unattend:
        print("...done!")
        print()
        print("Interpolating longitudinal data in space...")

    ### Interpolate Data
    t_l_m = interpolation(dist_T_L, t_L, dist_mod)
    temp_t0_m = interpolation(dist, temp_t0, dist_mod)

    ### Need to transpose discharge_m to make sure it has the same shape
    ### As discharge_m in Matlab.
    discharge_m = interpolation(dist_dis, discharge, dist_mod).transpose()

    width_m = interpolation(dist_stdim, width, dist_mod)
    depth_m = interpolation(dist_stdim, depth, dist_mod)
    depth_of_meas_m = interpolation(dist_bed, depth_of_meas, dist_mod)
    shade_m = interpolation(dist_shade, shade, dist_mod)
    vts_m = interpolation(dist_shade, vts, dist_mod)

    bed_temp_m = [0] * len(time_bed)
    for i in range(len(time_bed)):
        bed_temp_m[i] = interpolation(dist_bed, bed_temp[i], dist_mod)
    bed_temp_m = np.array(bed_temp_m).transpose()

    if not unattend:
        print("...done!")
        print()
        print("Interpolating temporal data in time...")

    ### Interpolate all data given through time so that there are
    ### Values at every step
    # checked!
    discharge_m = interpolation(time_dis, discharge_m, time_mod)

    ### Calculate width-depth-discharge relationship
    r = len(dist_mod)
    # checked!

    # theta = np.empty(r)
    theta = np.arctan((0.5 * width_m) / depth_m)

    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    # checked!
    dim_q = interpolation(dist_stdim, discharge_stdim, dist_mod)
    n_s = (
        (0.25 ** (2 / 3))
        * (2 ** (5 / 3))
        * (cos_theta ** (2 / 3))
        * (tan_theta ** (5 / 3))
        * (depth_m ** (8 / 3))
    ) / (2 * dim_q)

    discharge_m = discharge_m.transpose()

    # checked!
    depth_m = (
        (2 * n_s * discharge_m)
        / (
            (0.25 ** (2 / 3))
            * (2 ** (5 / 3))
            * (cos_theta ** (2 / 3))
            * (tan_theta ** (5 / 3))
        )
    ) ** (3 / 8)

    # checked!
    width_m = (
        2
        * tan_theta
        * (
            (
                (2 * n_s * discharge_m)
                / (
                    (0.25 ** (2 / 3))
                    * (2 ** (5 / 3))
                    * (cos_theta ** (2 / 3))
                    * (tan_theta ** (5 / 3))
                )
            )
            ** (3 / 8)
        )
    )

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
    temp_x0_dt = interpolation(time_temp, temp_x0, time_mod, "pchip")

    # checked!
    bed_temp_dt = [0] * r
    for i in range(r):
        bed_temp_dt[i] = interpolation(time_bed, bed_temp_m[i], time_mod)
    bed_temp_dt = np.array(bed_temp_dt)

    # checked!
    solar_rad_mat = np.array([solar_rad_dt] * r)
    air_temp_mat = np.array([air_temp_dt] * r)
    rel_hum_mat = np.array([rel_hum_dt] * r)
    wind_speed_mat = np.array([wind_speed_dt] * r)
    cl = np.array([c_dt] * r)

    if not unattend:
        print("...done!")
        print()

    ###############################################################
    # STEP 1: compute volumes of each reservoir (node) in the model
    # checked!
    if not unattend:
        print(
            "Computing volumes of nodes, discharge rates and groundwater inflow rates..."
        )
    volume = np.empty((r, timesteps))
    volume[0] = (dist_mod[1] - dist_mod[0]) * area_m[0]
    volume[1 : r - 1] = (
        area_m[1 : r - 1].transpose()
        * (
            ((dist_mod[2:] + dist_mod[1 : r - 1]) / 2)
            - ((dist_mod[1 : r - 1] + dist_mod[: r - 2]) / 2)
        )
    ).transpose()
    volume[r - 1] = area_m[r - 1] * (dist_mod[r - 1] - dist_mod[r - 2])

    ###############################################################
    # STEP 2: compute discharge rates at reservoir edges using linear
    # interpolation (n-values are upstream of node n values)
    # checked!
    q_half = np.empty((r + 1, timesteps))
    q_half[0] = (2 * discharge_m[0] - discharge_m[1] + discharge_m[0]) / 2
    q_half[1:r] = (discharge_m[1:r] + discharge_m[0 : r - 1]) / 2
    q_half[r] = (2 * discharge_m[r - 1] - discharge_m[r - 2] + discharge_m[r - 1]) / 2

    ###############################################################
    # STEP 3: compute lateral groundwater discharge rates to each node based on
    # longtudinal changes in streamflow
    # checked!
    q_l = q_half[1 : r + 1] - q_half[:r]

    ###############################################################
    # STEP 4: unit conversions so all discharge rates are in m3/min
    # note that all inputs are x are in m, T are in deg C, and Q or Q_L are in m3/s
    # checked!
    q_half_min = q_half * 60
    q_l_min = q_l * 60

    if not unattend:
        print("...done!")
        print()

    # method 2 creates overflow and the example_data uses method 1
    method = 1
    match method:
        case 1:
            ###############################################################
            # STEP 5: Calculate coefficients of the tridiagonal matrix (a, b, c)
            # and set coefficients at the boundaries. Use a, b and c to create the A
            # matrix.  Note that a, b and c are constant in time as long as Q,
            # volume, and Q_L are constant with time.
            # checked!
            double_volume = 2 * volume[:, 0]
            quad_volume = 4 * volume[:, 0]
            a = np.empty((r, timesteps))
            a[:, : timesteps - 1] = (
                (-dt * q_half_min[:r, 1:]).transpose() / quad_volume
            ).transpose()
            a[:, timesteps - 1] = (-dt * q_half_min[:r, timesteps - 1]) / quad_volume

            # checked!
            b = np.empty((r, timesteps))
            o = np.empty((r, timesteps))
            p = np.empty((r, timesteps))
            q = np.empty((r, timesteps))
            o[:, : timesteps - 1] = (
                (dt * q_half_min[:r, 1:]).transpose() / quad_volume
            ).transpose()
            p[:, : timesteps - 1] = (
                (dt * q_half_min[1:, 1:]).transpose() / quad_volume
            ).transpose()
            q[:, : timesteps - 1] = (
                (dt * q_l_min[:, : timesteps - 1]).transpose() / double_volume
            ).transpose()
            o[:, timesteps - 1] = (
                (dt * q_half_min[:r, timesteps - 1]).transpose() / quad_volume
            ).transpose()
            p[:, timesteps - 1] = (
                (dt * q_half_min[1:, timesteps - 1]).transpose() / quad_volume
            ).transpose()
            q[:, timesteps - 1] = (
                (dt * q_l_min[:r, timesteps - 1]).transpose() / double_volume
            ).transpose()
            b[:, : timesteps - 1] = (
                1
                + o[:, : timesteps - 1]
                - p[:, : timesteps - 1]
                + q[:, : timesteps - 1]
            )
            b[:, timesteps - 1] = (
                1 + o[:, timesteps - 1] - p[:, timesteps - 1] + q[:, timesteps - 1]
            )

            # checked!
            c = np.empty((r, timesteps))
            c[:, : timesteps - 1] = (
                (dt * q_half_min[1:, 1:]).transpose() / quad_volume
            ).transpose()
            c[:, timesteps - 1] = (dt * q_half_min[1:, timesteps - 1]) / quad_volume

            # all checked!
            a_c = ((-dt * q_half_min[:r]).transpose() / quad_volume).transpose()
            o_c = ((dt * q_half_min[:r]).transpose() / quad_volume).transpose()
            p_c = ((dt * q_half_min[1:, :]).transpose() / quad_volume).transpose()
            q_c = ((dt * q_l_min).transpose() / double_volume).transpose()
            b_c = (1 + o_c - p_c + q_c).transpose()
            c_c = ((dt * q_half_min[1:, :]).transpose() / quad_volume).transpose()

            ###############################################################
            # STEP 6: Calculate right hand side (d).
            # The values for d are temperature-dependent, so they change each time step.
            # Once d is computed, use that d value and the
            # matrix A to solve for the temperature for each time step.
            if not unattend:
                print(
                    "Computing d-values, heat fluxes and solving for stream temperatures..."
                )
            d = np.empty((r, timesteps))
            t = np.empty((r, timesteps))
            t[:, 0] = temp_t0_m
            heat_flux = np.empty((r, timesteps))
            shortwave = np.empty((r, timesteps))
            longwave = np.empty((r, timesteps))
            atm = np.empty((r, timesteps))
            back = np.empty((r, timesteps))
            land = np.empty((r, timesteps))
            latent = np.empty((r, timesteps))
            sensible = np.empty((r, timesteps))
            bed = np.empty((r, timesteps))

            (
                heat_flux[:, 0],
                shortwave[:, 0],
                longwave[:, 0],
                atm[:, 0],
                back[:, 0],
                land[:, 0],
                latent[:, 0],
                sensible[:, 0],
                bed[:, 0],
            ) = hflux_calculations.heatflux_calculations(
                input_data["settings"],
                solar_rad_mat[:, 0],
                air_temp_mat[:, 0],
                rel_hum_mat[:, 0],
                temp_t0_m,
                wind_speed_mat[:, 0],
                z,
                sed,
                bed_temp_dt[:, 0],
                depth_of_meas_m,
                shade_m,
                vts_m,
                cl[:, 0],
                sol_refl[0],
                wp_m[:r, 0],
                width_m[:, 0],
            )

            rho_water = 1000
            c_water = 4182

            g = 1 + a_c + c_c - q_c

            # Could be done better potentially, leave it for alpha for now
            k = np.empty((r, timesteps))
            for i in range(r):
                for j in range(timesteps):
                    if q_l_min[i, j] < 0:
                        k[i, j] = 0
                    else:
                        k[i, j] = (dt * q_l_min[i, j]) / (volume[i, 0])

            m = np.zeros(width_m.shape)
            d = np.zeros(width_m.shape)
            for i in range(timesteps - 1):
                m[:, i] = (
                    dt
                    * (width_m[:, i] * heat_flux[:, i])
                    / ((rho_water * c_water))
                    / area_m[:, i]
                )
                d[0, i] = (
                    (g[0, i] * t[0, i])
                    + (o_c[0, i] * temp_x0_dt[i])
                    - (p_c[0, i] * t[1, i])
                    + (k[0, i] * t_l_m[0])
                    + m[0, i]
                ) - (a_c[0, i] * temp_x0_dt[i + 1])
                d[r - 1, i] = (
                    (g[r - 1, i] * t[r - 1, i])
                    + (o_c[r - 1, i] * t[r - 2, i])
                    - (p_c[r - 1, i] * t[r - 1, i])
                    + (k[r - 1, i] * t_l_m[r - 1])
                    + m[r - 1, i]
                )
                d[1 : r - 1, i] = (
                    (g[1 : r - 1, i] * t[1 : r - 1, i])
                    + (o_c[1 : r - 1, i] * t[0 : r - 2, i])
                    - (p_c[1 : r - 1, i] * t[2:r, i])
                    + (k[1 : r - 1, i] * t_l_m[1 : r - 1])
                    + m[1 : r - 1, i]
                )

                A = np.zeros((r, r))
                for j in range(r - 1):
                    A[j + 1, j] = a[j + 1, i]
                    A[j, j] = b[j, i]
                    A[j, j + 1] = c[j, i]
                    A[r - 1, r - 1] = b[r - 1, i] + c[r - 1, i]

                A = csc_matrix(A)
                t[:, i + 1] = spsolve(A, d[:, i])

                (
                    heat_flux[:, i + 1],
                    shortwave[:, i + 1],
                    longwave[:, i + 1],
                    atm[:, i + 1],
                    back[:, i + 1],
                    land[:, i + 1],
                    latent[:, i + 1],
                    sensible[:, i + 1],
                    bed[:, i + 1],
                ) = hflux_calculations.heatflux_calculations(
                    input_data["settings"],
                    solar_rad_mat[:, i + 1],
                    air_temp_mat[:, i + 1],
                    rel_hum_mat[:, i + 1],
                    t[:, i + 1],
                    wind_speed_mat[:, i + 1],
                    z,
                    sed,
                    bed_temp_dt[:, i + 1],
                    depth_of_meas_m,
                    shade_m,
                    vts_m,
                    cl[:, i + 1],
                    sol_refl[i + 1],
                    wp_m[:, i + 1],
                    width_m[:, i + 1],
                )

        # Overflow error! Will come back to it for beta
        # case 2:
        #     t = np.zeros((r, timesteps))
        #     t[:,0] = temp_t0_m
        #     t[0,:] = temp_x0_dt
        #     t_k1 = np.zeros((r, timesteps))
        #     heat_flux = np.zeros((r, timesteps))

        #     heat_flux = np.empty((r, timesteps))
        #     heat_flux_k1 = np.empty((r, timesteps))
        #     heat_flux_k2 = np.empty((r, timesteps))
        #     shortwave = np.empty((r, timesteps))
        #     longwave = np.empty((r, timesteps))
        #     atm = np.empty((r, timesteps))
        #     back = np.empty((r, timesteps))
        #     land = np.empty((r, timesteps))
        #     latent = np.empty((r, timesteps))
        #     sensible = np.empty((r, timesteps))
        #     bed = np.empty((r, timesteps))
        #     u1 = np.empty((r, timesteps))
        #     v1 = np.empty((r, timesteps))
        #     s1 = np.empty((r, timesteps))
        #     m1 = np.empty((r, timesteps))
        #     k1 = np.empty((r, timesteps))
        #     u2 = np.empty((r, timesteps))
        #     v2 = np.empty((r, timesteps))
        #     s2 = np.empty((r, timesteps))
        #     m2 = np.empty((r, timesteps))
        #     k2 = np.empty((r, timesteps))

        #     heat_flux[:, 0], shortwave[:, 0], longwave[:, 0], atm[:, 0], back[:, 0], land[:, 0], latent[:, 0], sensible[:, 0], bed[:, 0] = hflux_flux(input_data["settings"], solar_rad_mat[:, 0],
        #                                             air_temp_mat[:, 0], rel_hum_mat[:, 0], temp_t0_m,
        #                                             wind_speed_mat[:, 0], z, sed, bed_temp_dt[:, 0],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, 0], sol_refl[0], wp_m[:r, 0], width_m[:, 0])

        #     for i in range(timesteps - 1):
        #         heat_flux_k1[:, i], _, _, _, _, _, _, _, _ = hflux_flux(input_data["settings"], solar_rad_mat[:, i],
        #                                             air_temp_mat[:, i], rel_hum_mat[:, i], t[:, i],
        #                                             wind_speed_mat[:, i], z, sed, bed_temp_dt[:, i],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, i], sol_refl[i], wp_m[:, i], width_m[:, i])

        #         for j in range(1, r - 1):
        #             u1[j, i] = (q_half_min[j,i] / volume[j, 0]) * (.5 * t[j - 1, i] - .5 * t[j, i])
        #             v1[j, i]=(q_half_min[j + 1, i] / volume[j, 0]) * (0.5 * t[j, i] - 0.5 * t[j + 1,i])
        #             s1[j, i]=(q_l_min[j, i] / volume[j, 0]) * (t_l_m[j] - t[j, i])

        #             rho_water = 1000
        #             c_water = 4182
        #             m1[j, i] = (width_m[j,i] * heat_flux_k1[j,i]) / ((rho_water*c_water)) / area_m[j, i]
        #             k1[j, i] = u1[j, i] + v1[j, i] + s1[j, i] + m1[j, i]

        #         u1[r - 1, i] = (q_half_min[r - 1,i] / volume[r - 1, 0]) * (.5 * t[r - 2, i] - .5 * t[r - 1, i])
        #         v1[r - 1, i]=(q_half_min[r, i] / volume[r - 1, 0]) * (0.5 * t[r - 1, i] - 0.5 * t[r - 1,i])
        #         s1[r - 1, i]=(q_l_min[r - 1, i] / volume[r - 1, 0]) * (t_l_m[r - 1] - t[r - 1, i])

        #         m1[r - 1, i] = (width_m[r - 1,i] * heat_flux_k1[r - 1,i]) / ((rho_water*c_water)) / area_m[r - 1, i]
        #         k1[r - 1, i] = u1[r - 1, i] + v1[r - 1, i] + s1[r - 1, i] + m1[r - 1, i]

        #         # Calculate temp based on k1
        #         t_k1[0, i] = temp_x0_dt[i]
        #         for j in range(1, r):
        #             t_k1[j, i] = t[j, i] + (dt * k1[j, i])

        #         heat_flux_k2[:, i], _, _, _, _, _, _, _, _ = hflux_flux(input_data["settings"], solar_rad_mat[:, i],
        #                                             air_temp_mat[:, i], rel_hum_mat[:, i], t_k1[:, i],
        #                                             wind_speed_mat[:, i], z, sed, bed_temp_dt[:, i],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, i], sol_refl[i], wp_m[:, i], width_m[:, i])

        #         for j in range(1, r - 1):
        #             u2[j, i] = (q_half_min[j, i + 1] / volume[j, 0]) * (.5 * t_k1[j - 1, i] - .5 * t_k1[j, i])
        #             v2[j, i]=(q_half_min[j + 1, i + 1] / volume[j, 0]) * (0.5 * t_k1[j, i] - 0.5 * t_k1[j + 1,i])
        #             s2[j, i]=(q_l_min[j, i + 1] / volume[j, 0]) * (t_l_m[j] - t_k1[j, i])

        #             m2[j, i] = (width_m[j,i] * heat_flux_k2[j,i]) / ((rho_water*c_water)) / area_m[j, i]
        #             k2[j, i] = u2[j, i] + v2[j, i] + s2[j, i] + m2[j, i]

        #         u2[r - 1, i] = (q_half_min[r - 1, i + 1] / volume[r - 1, 0]) * (.5 * t_k1[r - 2, i] - .5 * t_k1[r - 1, i])
        #         v2[r - 1, i]=(q_half_min[r, i] / volume[r - 1, 0]) * (0.5 * t_k1[r - 1, i] - 0.5 * t_k1[r - 1,i])
        #         s2[r - 1, i]=(q_l_min[r - 1, i] / volume[r - 1, 0]) * (t_l_m[r - 1] - t_k1[r - 1, i])

        #         m2[r - 1, i] = (width_m[r - 1,i] * heat_flux_k2[r - 1,i]) / ((rho_water*c_water)) / area_m[r - 1, i]
        #         k2[r - 1, i] = u2[r - 1, i] + v2[r - 1, i] + s2[r - 1, i] + m2[r - 1, i]

        #         for j in range(1, r):
        #             t[j, i + 1] = t[j, i] + (dt * (.5 * k1[j, i]) + .5 * k2[j, i])

        #         heat_flux[:,i+1], shortwave[:,i+1], longwave[:,i + 1], atm[:,i + 1], back[:,i + 1], land[:,i + 1], latent[:,i + 1],sensible[:,i + 1], bed[:,i + 1] = hflux_flux(input_data["settings"],
        #             solar_rad_mat[:,i+1],air_temp_mat[:,i+1],
        #             rel_hum_mat[:,i+1],t[:,i+1],wind_speed_mat[:,i+1],z,
        #             sed,bed_temp_dt[:,i+1],depth_of_meas_m,
        #             shade_m,vts_m,cl[:,i+1],sol_refl[i + 1], wp_m[:,i+1], width_m[:,i+1])
        #         print(i)

        #     print(heat_flux)

    if not unattend:
        print("...done!")
        print()

    # 2D Plot of stream temperature
    # Create a figure
    if not unattend:
        fig, ax = plt.subplots()

        pdf_path = os.path.join(os.getcwd(), "Results", "PDFs", "hflux.pdf")
        plots_pdf = PdfPages(pdf_path)

        # ax.imshow() - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
        cax = ax.imshow(
            t,
            aspect="auto",
            cmap="jet",
            origin="lower",
            extent=[
                np.min(time_mod),
                np.max(time_mod),
                np.min(dist_mod),
                np.max(dist_mod),
            ],
        )

        # Add a colorbar with label
        # plt.colorbar() - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        cbar = plt.colorbar(cax)
        cbar.set_label("Temperature (°C)", fontsize=12, fontweight="bold")

        # Set title and labels
        plot_title = "Modeled Stream Temperature"
        xlab = "Time (min)"
        ylab = "Distance (m)"
        ax.set_title(plot_title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)

        # ax.invert_yaxis() - https://www.geeksforgeeks.org/how-to-reverse-axes-in-matplotlib/
        ax.invert_yaxis()

        # 3D plot of stream temperature
        # fig = plt.figure()

        # # Create a 3D axis
        # ax = fig.add_subplot(111, projection="3d")

        # # Create a surface plot
        # # Make x, y axis take different length - https://stackoverflow.com/questions/46607106/python-3d-plot-with-different-array-sizes
        # time_mod_sized, dist_mod_sized = np.meshgrid(time_mod, dist_mod)
        # # ax.plot_surface() - https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html#plot-surface-x-y-z
        # surface = ax.plot_surface(dist_mod_sized, time_mod_sized, t, cmap="jet")

        # # Add a colorbar with label
        # cbar = fig.colorbar(surface)
        # cbar.set_label("Temperature (°C)", fontsize=11, fontweight="bold")

        # print()
        # print("INITIALZ")
        # # Set title and labels
        plot_title = "Modeled Stream Temperature"
        ylab = "Time (min)"
        xlab = "Distance (m)"
        zlab = "Temp (°C)"
        cbar = "Temperature (°C)"
        # ax.set_title(plot_title, fontsize=12, fontweight="bold")
        # ax.set_ylabel(ylab, fontsize=11)
        # ax.set_xlabel(xlab, fontsize=11)
        # ax.set_zlabel(zlab, fontsize=11)
        # ax.invert_xaxis()
        p = plc()
        fig = p.make3dplot(time_mod, dist_mod, t, xlab, ylab, zlab, cbar, plot_title)
        # fig.show()
        # Plot of heat fluxes
        fig = plt.figure()

        # Subplot 1
        plt.subplot(3, 2, 1)

        x = time_mod
        y = np.mean(heat_flux / 60, axis=0)
        marker = "k"
        title = "Total Heat Flux"
        ylabel = "Energy Flux (W/m^2)"
        xlabel = "X axis"
        axis = [
            np.min(time_mod),
            np.max(time_mod),
            np.min(np.mean(heat_flux / 60, axis=0)),
            np.max(np.mean(heat_flux / 60, axis=0)),
        ]
        fig = p.make_single_plot(x, y, xlabel, ylabel, title, marker=marker, axis=axis)
        # fig.show()
        # input("Press enter to close.")
        # sys.exit()
        # plt.plot(time_mod, np.mean(heat_flux / 60, axis=0), "k")
        # plt.title("Total Heat Flux", fontweight="bold")
        # plt.ylabel("Energy Flux (W/m^2)", fontweight="bold")
        # plt.axis(
        #     [
        #         np.min(time_mod),
        #         np.max(time_mod),
        #         np.min(np.mean(heat_flux / 60, axis=0)),
        #         np.max(np.mean(heat_flux / 60, axis=0)),
        #     ]
        # )
        # input("WAITING ON INPUT...")
        # sys.exit()

        # Subplot 2
        plt.subplot(3, 2, 2)
        plt.plot(time_mod, np.mean(shortwave, axis=0), "r")
        plt.title("Shortwave Radiation", fontweight="bold")
        plt.axis(
            [
                np.min(time_mod),
                np.max(time_mod),
                np.min(np.mean(shortwave, axis=0)),
                np.max(np.mean(shortwave, axis=0)),
            ]
        )

        # Subplot 3
        plt.subplot(3, 2, 3)
        plt.plot(time_mod, np.mean(longwave, axis=0), "b")
        plt.title("Longwave Radiation", fontweight="bold")
        plt.ylabel("Energy Flux (W/m^2)", fontweight="bold")
        plt.axis(
            [
                np.min(time_mod),
                np.max(time_mod),
                np.min(np.mean(longwave, axis=0)),
                np.max(np.mean(longwave, axis=0)),
            ]
        )

        # Subplot 4
        plt.subplot(3, 2, 4)
        plt.plot(time_mod, np.mean(latent, axis=0), "g")
        plt.title("Latent Heat Flux", fontweight="bold")
        plt.axis(
            [
                np.min(time_mod),
                np.max(time_mod),
                np.min(np.mean(latent, axis=0)),
                np.max(np.mean(latent, axis=0)),
            ]
        )

        # Subplot 5
        plt.subplot(3, 2, 5)
        plt.plot(time_mod, np.mean(bed, axis=0), "m")
        plt.title("Bed Conduction", fontweight="bold")
        plt.xlabel("Time (min)", fontweight="bold")
        plt.ylabel("Energy Flux (W/m^2)", fontweight="bold")
        plt.axis(
            [
                np.min(time_mod),
                np.max(time_mod),
                np.min(np.mean(bed, axis=0)),
                np.max(np.mean(bed, axis=0)),
            ]
        )

        # Subplot 6
        plt.subplot(3, 2, 6)
        plt.plot(time_mod, np.mean(sensible, axis=0), "y")
        plt.title("Sensible Heat Flux", fontweight="bold")
        plt.xlabel("Time (min)", fontweight="bold")
        plt.axis(
            [
                np.min(time_mod),
                np.max(time_mod),
                np.min(np.mean(sensible, axis=0)),
                np.max(np.mean(sensible, axis=0)),
            ]
        )

        # to avoid labels overlapping
        # cite: https://saturncloud.io/blog/how-to-improve-label-placement-for-matplotlib-scatter-chart/#:~:text=Matplotlib%20provides%20a%20feature%20called,the%20labels%20to%20minimize%20overlap.
        plt.tight_layout()

        # Plot of heat fluxes: comparison
        plt.figure()

        # Plot data
        plt.plot(
            time_mod, np.mean(heat_flux / 60, axis=0), "k", label="Total Heat Flux"
        )
        plt.plot(time_mod, np.mean(shortwave, axis=0), "r", label="Solar Radiation")
        plt.plot(time_mod, np.mean(longwave, axis=0), "b", label="Longwave Radiation")
        plt.plot(time_mod, np.mean(latent, axis=0), "g", label="Latent Heat Flux")
        plt.plot(time_mod, np.mean(bed, axis=0), "c", label="Streambed Conduction")
        plt.plot(time_mod, np.mean(sensible, axis=0), "m", label="Sensible Heat Flux")

        # Set axis properties
        plt.xlim([np.min(time_mod), np.max(time_mod)])

        # Add title and labels
        plt.title("Energy Fluxes", fontsize=12, fontweight="bold")
        plt.xlabel("Time (min)", fontsize=11, fontweight="bold")
        plt.ylabel("Energy Flux (W/m^2)", fontsize=11, fontweight="bold")

        # Add legend
        plt.legend(loc="best")

        # CITE: https://www.geeksforgeeks.org/save-multiple-matplotlib-figures-in-single-pdf-file-using-python/
        # get_fignums Return list of existing
        # figure numbers
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        # iterating over the numbers in list
        for fig in figs:
            # and saving the files
            fig.savefig(plots_pdf, format="pdf")

        plots_pdf.close()
        plt.close("all")

    if method == 1:
        matrix_data = {}
        matrix_data["a"] = a
        matrix_data["b"] = b
        matrix_data["c"] = c
        matrix_data["A"] = A
        matrix_data["o"] = o
        matrix_data["p"] = p
        matrix_data["q"] = q
        matrix_data["g"] = g
        matrix_data["k"] = k
        matrix_data["m"] = m
        matrix_data["d"] = d

        matrix_data["a_c"] = a_c
        matrix_data["b_c"] = b_c
        matrix_data["c_c"] = c_c
        matrix_data["o_c"] = o_c
        matrix_data["p_c"] = p_c
        matrix_data["q_c"] = q_c
    # will come back to case2 after alpha!
    # else:
    #     matrix_data["u1"] = u1
    #     matrix_data["v1"] = v1
    #     matrix_data["s1"] = s1
    #     matrix_data["m1"] = m1
    #     matrix_data["k1"] = k1
    #     matrix_data["u2"] = u2
    #     matrix_data["v2"] = v2
    #     matrix_data["s2"] = s2
    #     matrix_data["m2"] = m2
    #     matrix_data["k2"] = k2

    node_data = {}
    node_data["v"] = volume
    node_data["Q"] = q_half_min
    node_data["ql"] = q_l_min
    node_data["width"] = width_m
    node_data["area"] = area_m

    flux_data = {}
    flux_data["heatflux"] = heat_flux / 60
    flux_data["solarflux"] = shortwave
    flux_data["solar_refl"] = sol_refl
    flux_data["long"] = longwave
    flux_data["atmflux"] = atm
    flux_data["landflux"] = land
    flux_data["backrad"] = back
    flux_data["evap"] = latent
    flux_data["sensible"] = sensible
    flux_data["conduction"] = bed

    return t, matrix_data, node_data, flux_data
