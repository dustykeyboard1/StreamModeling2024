import numpy as np
import os
import sys
import math
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities import Input_reader
from src.Heat_Flux.hflux_bed_sed import hflux_bed_sed
from src.Heat_Flux.hflux_shortwave_refl import hflux_shortwave_relf
from src.Heat_Flux.hflux_flux import hflux_flux
from src.Utilities.interpolation import interpolation
from src.Heat_Flux.heatflux_calculations import HeatFluxCalculations

# probably don't need this
from Python.src.Utilities.data_table_class import DataTable



class HeatFlux:
    def __init__(self, input_data):
        self.data_table = input_data
        self.hflux_calculations = HeatFluxCalculations()
        self.timesteps = len(input_data.time_mod)
        self.r = len(self.data_table.dist_mod)
        self.dt = max(input_data.time_mod) / (self.timesteps - 1)

    def _calculate_hflux_bed_sed(self):
        return self.hflux_calculations.hflux_bed_sed(self.input_data.sed_type, self.input_data.dist_bed, self.input_data.dist_mod)
    
    def _calculate_sol_refl(self):
        return self.hflux_calculations.hflux_shortwave_refl(self.input_data.year, self.input_data.month, self.input_data.day, self.input_data.hour, self.input_data.minute, self.input_data.lat, self.input_data.lon, self.input_data.t_zone, self.input_data.time_met, self.input_data.time_mod)

    def _interpolate_data(self):
        ### Interpolate Data
        t_l_m = interpolation(self.data_table.dist_T_L, self.data_table.t_L, self.data_table.dist_mod)
        temp_t0_m = interpolation(self.data_table.dist, self.data_table.temp_t0, self.data_table.dist_mod)
        width_m = interpolation(self.data_table.dist_stdim, self.data_table.width, self.data_table.dist_mod)
        depth_m = interpolation(self.data_table.dist_stdim, self.data_table.depth, self.data_table.dist_mod)
        depth_of_meas_m = interpolation(self.data_table.dist_bed, self.data_table.depth_of_meas,self.data_table.dist_mod)
        shade_m = interpolation(self.data_table.dist_shade, self.data_table.shade, self.data_table.dist_mod)
        vts_m = interpolation(self.data_table.dist_shade, self.data_table.vts, self.data_table.dist_mod)

        time_bed_length = len(self.data_table.time_bed)
        bed_temp_m = [0] * time_bed_length
        for i in range(time_bed_length):
            bed_temp_m[i] = interpolation(self.data_table.dist_bed, self.data_table.bed_temp[i], self.data_table.dist_mod)
        bed_temp_m = np.array(bed_temp_m).transpose()

        solar_rad_dt = interpolation(self.data_table.time_met, self.data_table.solar_rad_in, self.data_table.time_mod)
        air_temp_dt = interpolation(self.data_table.time_met, self.data_table.air_temp_in, self.data_table.time_mod)
        rel_hum_dt = interpolation(self.data_table.time_met, self.data_table.rel_hum_in, self.data_table.time_mod)
        wind_speed_dt = interpolation(self.data_table.time_met, self.data_table.wind_speed_in, self.data_table.time_mod)
        c_dt = interpolation(self.data_table.time_cloud, self.data_table.c_in, self.data_table.time_mod)
        temp_x0_dt = interpolation(self.data_table.time_temp, self.data_table.temp_x0, self.data_table.time_mod, "pchip")

        # checked!
        bed_temp_dt = [0] * self.r
        for i in range(self.r):
            bed_temp_dt[i] = interpolation(self.data_table.time_bed, bed_temp_m[i], self.data_table.time_mod)
        bed_temp_dt = np.array(bed_temp_dt)

        self.theta = np.arctan((0.5 * width_m) / depth_m)
        self.dim_q = interpolation(self.data_table.dist_stdim, self.data_table.discharge_stdim, self.data_table.dist_mod)

        return t_l_m, temp_t0_m, width_m, depth_m, depth_of_meas_m, shade_m, vts_m, bed_temp_m, solar_rad_dt, air_temp_dt, rel_hum_dt, wind_speed_dt, c_dt, temp_x0_dt, bed_temp_dt
    
    def _interpolate_discharge_m(self):
        ### Need to transpose discharge_m to make sure it has the same shape
        ### As discharge_m in Matlab.
        discharge_m = interpolation(self.data_table.dist_dis, self.data_table.discharge, self.data_table.dist_mod).transpose()
        discharge_m = interpolation(self.data_table.time_dis, discharge_m, self.data_table.time_mod)

        return discharge_m

    def _calculate_width_depth_discharge_relationship(self):
        ### Calculate width-depth-discharge relationship
        discharge_m = self.discharge_m.transpose()

        cos_theta = np.cos(self.theta)
        tan_theta = np.tan(self.theta)

        n_s = (
            (0.25 ** (2 / 3))
            * (2 ** (5 / 3))
            * (cos_theta ** (2 / 3))
            * (tan_theta ** (5 / 3))
            * (depth_m ** (8 / 3))
        ) / (2 * self.dim_q)

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

        self.width_m = width_m
        self.area_m = area_m
        self.wp_m = wp_m
    
    def _calculate_reservoir_volumes(self):
        if not self.data_table.unattend:
            print('Computing volumes of nodes, discharge rates and groundwater inflow rates...')
        volume = np.empty((self.r, self.timesteps))
        volume[0] = (self.data_table.dist_mod[1] - self.data_table.dist_mod[0]) * self.area_m[0]
        volume[1: self.r-1] = (self.area_m[1:self.r-1].transpose() * 
                        (((self.data_table.dist_mod[2:] + self.data_table.dist_mod[1:self.r-1]) / 2) - 
                        ((self.data_table.dist_mod[1:self.r-1] + self.data_table.dist_mod[:self.r-2]) / 2))).transpose()
        volume[self.r-1] = self.area_m[self.r-1] * (self.data_table.dist_mod[self.r-1]-self.data_table.dist_mod[self.r-2])

        return volume
    
    # have to change the magic number 60 later, it is to convert unit
    def _linearly_calculate_reservoir_edges_discharge_rates(self):
        q_half = np.empty((self.r+1, self.timesteps))
        q_half[0] = (2 * self.discharge_m[0] - self.discharge_m[1] + self.discharge_m[0]) / 2
        q_half[1:self.r] = (self.discharge_m[1:self.r]+self.discharge_m[0:self.r-1]) / 2
        q_half[self.r] = (2 * self.discharge_m[self.r-1] - self.discharge_m[self.r-2] + self.discharge_m[self.r-1]) / 2

        return q_half * 60
    
    def _lateral_groundwater_discharge_rates_through_longtudinal_changes(self):
        q_half = self._linearly_calculate_reservoir_edges_discharge_rates()
        q_l = q_half[1:self.r + 1] - q_half[:self.r]

        return q_l * 60
    
    def crank_nicolson_method():
        pass

    def runge_kutta_method():
        pass

