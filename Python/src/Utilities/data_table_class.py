import pandas as pd
import numpy as np
import os
from copy import deepcopy

class DataTable:
    def __init__(self, file_name = ""):
        if file_name != "":
            input_data = self._input_reader(file_name)

            # initialize variables
            self.input_data = input_data

            self.settings = input_data["settings"]
            self.unattend = bool(self.settings[0][4])
            self.method = self.settings[0][0]

            if not self.unattend:
                print('Assigning variable names...')
            
            self.temp = input_data['temp']

            self.time_mod_raw = input_data["time_mod"]

            self.dist_mod_raw = input_data["dist_mod"]

            self.temp_x0_data = input_data["temp_x0_data"]

            self.temp_t0_data = input_data["temp_t0_data"]

            self.dim_data = input_data["dim_data"]
            # # area = input_data["dim_data"][1]  p.s. commented out in the original Matlab file
            
            self.dis_data = input_data["dis_data"]

            self.time_dis_raw = input_data["time_dis"]

            self.t_l_data = input_data["T_L_data"]

            self.met_data = input_data["met_data"]

            self.bed_data1 = input_data["bed_data1"]

            self.bed_data2 = input_data["bed_data2"]

            self.sed_type_raw = input_data["sed_type"]

            self.shade_data = input_data["shade_data"]

            self.cloud_data = input_data["cloud_data"]

            self.site_info = input_data["site_info"]

            self._format_checking_after_reading()

    @property
    def time_mod(self):
        return self.time_mod_raw[0]
    
    @property
    def dist_mod(self):
        return self.dist_mod_raw[0]

    @property
    def time_temp(self):
        return self.temp_x0_data[0]

    @property
    def temp_x0(self):
        return self.temp_x0_data[1]

    @property
    def dist_temp(self):
        return self.temp_t0_data[0]

    @property
    def temp_t0(self):
        return self.temp_t0_data[1]

    @property
    def dist_stdim(self):
        return self.dim_data[0]
    
    @property
    def width(self):
        return self.dim_data[2]

    @property
    def depth(self):
        return self.dim_data[3]
    
    @property
    def discharge_stdim(self):
        return self.dim_data[4]
    
    @property
    def dist_dis(self):
        return self.dis_data[0]
    
    @property
    def discharge(self):
        return self.dis_data[1:]

    @property
    def time_dis(self):
        return self.time_dis_raw[0]

    @property
    def dist_T_L(self):
        return self.t_l_data[0]

    @property
    def t_L(self):
        return self.t_l_data[1]  

    @property
    def year(self):
        return self.met_data[0]
    
    @property
    def month(self):
        return self.met_data[1]
    
    @property
    def day(self):
        return self.met_data[2]
    
    @property
    def hour(self):
        return self.met_data[3]

    @property
    def minute(self):
        return self.met_data[4]

    @property
    def time_met(self):
        return self.met_data[5]
    
    @property
    def solar_rad_in(self):
        return self.met_data[6]

    @property
    def air_temp_in(self):
        return self.met_data[7]
    
    @property
    def rel_hum_in(self):
        return self.met_data[8]
    
    @property
    def wind_speed_in(self):
        return self.met_data[9]

    @property
    def dist_bed(self):
        return self.bed_data1[0]

    @property
    def depth_of_meas(self):
        return self.bed_data1[1]
    
    @property
    def time_bed(self):
        return np.array([self.bed_data2[0, 0], self.bed_data2[1, 0]])
    
    @property
    def bed_temp(self):
        return self.bed_data2[0:, 1:]
    
    @property
    def sed_type(self):
        return self.sed_type_raw[0]
    
    @property
    def dist_shade(self):
        return self.shade_data[0]
    
    @property
    def shade(self):
        return self.shade_data[1]
    
    @property
    def vts(self):
        return self.shade_data[2]
    
    @property
    def time_cloud(self):
        return self.cloud_data[0]

    @property
    def c_in(self):
        return self.cloud_data[1]
    
    @property
    def lat(self):
        return self.site_info[0, 0]

    @property
    def lon(self):
        return self.site_info[0, 1]

    @property
    def t_zone(self):
        return self.site_info[0, 2]
    
    @property
    def z(self):
        return self.site_info[0, 3]
    
    def modify_data_table(self, sheet_to_change_name, new_value):
        modified_data_table = deepcopy(self)
        setattr(modified_data_table, sheet_to_change_name, new_value)

        return modified_data_table

    def _input_reader(self, file_name):
        data = pd.ExcelFile(file_name)
        sheet_names = data.sheet_names
        sheet_num = len(sheet_names)

        input_data = {}
        for i in range(0, sheet_num):
            d = pd.read_excel(data, sheet_names[i]).values.transpose()
            self._format_checking_during_reading(len(d), sheet_names[i])
            input_data[sheet_names[i]] = d

        return input_data

    def _format_checking_during_reading(self, colNum, sheet_name):
        if sheet_name in ("temp_x0_data", "temp_t0_data", "T_L_data",
                        "bed_data1", "cloud_data") and colNum != 2:
            print(sheet_name + " must contain 2 columns of data!")
        elif sheet_name == "dim_data" and colNum != 5:
            print(sheet_name + " must contain 5 columns of data!")
        elif sheet_name == "met_data" and colNum != 10:
            print(sheet_name + " must contain 10 columns of data!")
        elif sheet_name == "shade_data" and colNum != 3:
            print(sheet_name + " must contain 3 columns of data!")
        elif sheet_name in ("site_info", "time_mod", "dist_mod", "sed_type") and colNum != 1: #what is a cell array? Column vector?
            print(sheet_name + " must contain 1 columns of data!")

    def _format_checking_after_reading(self):
        # #Check for Boolean. 
        if not self.unattend:
            print("Checking input arguments...")

        # Type check variables to ensure they are column vectors by checking number of dimnesions and shape. 
        # ndim - https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
        # shape - https://numpy.org/doc/stable/reference/generated/numpy.shape.html
        if self.time_mod.ndim != 1:
            raise TypeError("Time_m must be a column vector")
        if self.dist_mod.ndim != 1:
            raise TypeError("Dist_m must be a column vector")
        if self.time_temp.ndim != 1:
            raise TypeError("Time_temp must be a column vector")
        if self.dist_temp.ndim != 1:
            raise TypeError("Dist_temp must be a column vector.")

        # Ensure temp_mod and temp are ndarrays.
        # Isinstance - https://docs.python.org/3/library/functions.html#isinstance
        # if not isinstance(self.temp_mod, np.ndarray): 
        #     raise TypeError("Temp_mod must be a numpy array representing a matrix.")
        if not isinstance(self.temp, np.ndarray):
            raise TypeError("Temp must be a numpy array representing a matrix.")
        
        if not self.unattend:
            print('...Done!')


