"""
Author: Violet Shi
File: data_table_class.py
Date: 11-05-2023
Functionality: Construct a dataTable class for reading and storing data from the input file.
"""

import pandas as pd
import numpy as np
from copy import deepcopy

class DataTable:
    def __init__(self, file_name):
        """
        Create a DataTable instance that read, check, and store data from the input file

        Args: file_name (string): The name of the input file
        """
        # read in the input file
        input_data = self._input_reader(file_name)

        # storing all data sheets as attribute to prepare for automatically updating column variables later
        self.settings = input_data["settings"]
        self.temp = input_data["temp"]['temp']
        self.time_mod_raw = input_data["time_mod"]
        self.dist_mod_raw = input_data["dist_mod"]
        self.temp_x0_data = input_data["temp_x0_data"]
        self.temp_t0_data = input_data["temp_t0_data"]
        self.dim_data = input_data["dim_data"]  
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

        # check if the format of the data is correct after reading in the input file
        self._format_checking_after_reading()

    @property
    def output_suppression(self):
        """
        Initialize variable output_suppression
        """
        return bool(self.settings["output suppression"])

    @output_suppression.setter
    def output_suppression(self, value):
        """
        Set output_suppression to value

        Args: value (bool): whether to print outputs or not
        """
        self.settings["output suppression"] = value

    @property
    def shortwave_radiation_method(self):
        """
        Initialize variable shortwave_radiation_method
        """
        return self.settings["shortwave radiation method"]

    @shortwave_radiation_method.setter
    def shortwave_radiation_method(self, value):
        """
        Set shortwave_radiation_method to value

        Args: value (int): 1 for Crank-Nicolson; 2 for second order Runge-Kutta
        """
        self.settings["shortwave radiation method"] = value

    @property
    def time_mod(self):
        """
        Initialize variable time_mod
        """
        return self.time_mod_raw["Time"]
    
    @property
    def dist_mod(self):
        """
        Initialize variable dist_mod
        """
        return self.dist_mod_raw["Distance"]

    @property
    def time_temp(self):
        """
        Initialize variable time_temp
        """
        return self.temp_x0_data["Time"]

    @property
    def temp_x0(self):
        """
        Initialize variable temp_x0
        """
        return self.temp_x0_data["Temperature"]

    @property
    def dist_temp(self):
        """
        Initialize variable dist_temp
        """
        return self.temp_t0_data["Distance"]

    @property
    def temp_t0(self):
        """
        Initialize variable temp_t0
        """
        return self.temp_t0_data["Temperature"]

    @property
    def dist_stdim(self):
        """
        Initialize variable dist_stdim
        """
        return self.dim_data["Distance"]
    
    # p.s. this is commented out in the original Matlab file
    # @property
    # def area(self):
    #     return self.dim_data["Area"]

    @property
    def width(self):
        """
        Initialize variable width
        """
        return self.dim_data["Width"]

    @property
    def depth(self):
        """
        Initialize variable depth
        """
        return self.dim_data["Depth"]
    
    @property
    def discharge_stdim(self):
        """
        Initialize variable discharge_stdim
        """
        return self.dim_data["Discharge"]
    
    @property
    def dist_dis(self):
        """
        Initialize variable discharge_stdim
        """
        return self.dis_data["Distance"]
    
    @property
    def discharge(self):
        """
        Initialize variable discharge
        """
        return self.dis_data["Q"]

    @property
    def time_dis(self):
        """
        Initialize variable time_dis
        """
        return self.time_dis_raw["Time"]

    @property
    def dist_T_L(self):
        """
        Initialize variable dist_T_L
        """
        return self.t_l_data["Distance"]

    @property
    def t_L(self):
        """
        Initialize variable t_L
        """
        return self.t_l_data["Temperature"]

    @property
    def year(self):
        """
        Initialize variable year
        """
        return self.met_data["Year"]
    
    @property
    def month(self):
        """
        Initialize variable month
        """
        return self.met_data["Month"]
    
    @property
    def day(self):
        """
        Initialize variable day
        """
        return self.met_data["Day"]
    
    @property
    def hour(self):
        """
        Initialize variable hour
        """
        return self.met_data["Hour"]

    @property
    def minute(self):
        """
        Initialize variable minute
        """
        return self.met_data["Minute"]

    @property
    def time_met(self):
        """
        Initialize variable time_met
        """
        return self.met_data["Time"]
    
    @property
    def solar_rad_in(self):
        """
        Initialize variable solar_rad_in
        """
        return self.met_data["Shortwave Radiation"]

    @property
    def air_temp_in(self):
        """
        Initialize variable air_temp_in
        """
        return self.met_data["Air Temperature"]
    
    @property
    def rel_hum_in(self):
        """
        Initialize variable rel_hum_in
        """
        return self.met_data["Relative Humidity"]
    
    @property
    def wind_speed_in(self):
        """
        Initialize variable wind_speed_in
        """
        return self.met_data["Wind Speed"]

    @property
    def dist_bed(self):
        """
        Initialize variable dist_bed
        """
        return self.bed_data1["Distance"]

    @property
    def depth_of_meas(self):
        """
        Initialize variable depth_of_meas
        """
        return self.bed_data1["Depth of Measurement"]
    
    @property
    def time_bed(self):
        """
        Initialize variable time_bed
        """
        return np.array([self.bed_data2["bed_data2"][0, 0], self.bed_data2["bed_data2"][1, 0]])
    
    @property
    def bed_temp(self):
        """
        Initialize variable bed_temp
        """
        return self.bed_data2["bed_data2"][0:, 1:]
    
    @property
    def sed_type(self):
        """
        Initialize variable sed_type
        """
        return self.sed_type_raw['sed_type'][0]
    
    @property
    def dist_shade(self):
        """
        Initialize variable dist_shade
        """
        return self.shade_data["Distance"]
    
    @property
    def shade(self):
        """
        Initialize variable shade
        """
        return self.shade_data["Shade "]
    
    @property
    def vts(self):
        """
        Initialize variable vts
        """
        return self.shade_data["View to Sky"]
    
    @property
    def time_cloud(self):
        """
        Initialize variable time_cloud
        """
        return self.cloud_data["Time"]

    @property
    def c_in(self):
        """
        Initialize variable c_in
        """
        return self.cloud_data["Cloud Cover"]
    
    @property
    def lat(self):
        """
        Initialize variable lat
        """
        return self.site_info["site_info"][0, 0]

    @property
    def lon(self):
        """
        Initialize variable lon
        """
        return self.site_info["site_info"][0, 1]

    @property
    def t_zone(self):
        """
        Initialize variable t_zone
        """
        return self.site_info["site_info"][0, 2]
    
    @property
    def z(self):
        """
        Initialize variable z
        """
        return self.site_info["site_info"][0, 3]
    
    def modify_data_table(self, sheet_to_change_name, new_value):
        """
        Create a new instance of data_table that's based on the original data_table with some sheets changed 

        Args: sheet_to_change_name (string or a list of strings): the names of the sheets to be changed 
              new_value (ndarrays or a list of ndarrays): new values of the sheets to be changed 

        Returns: modified_data_table (DataTable): new modified data table
        """
        modified_data_table = deepcopy(self)
        if type(sheet_to_change_name) == list:
            for i in range(len(sheet_to_change_name)):
                setattr(modified_data_table, sheet_to_change_name[i], new_value[i])
        else:
            setattr(modified_data_table, sheet_to_change_name, new_value)

        return modified_data_table

    def _slice_column_name(self, col_name, target):
        """
        Get the name of column without units/spaces

        Args: col_name (string): the raw name of the column to be sliced
              target (string): the unwanted part of the name
        """
        end_index = col_name.find(target)
        if end_index != -1:
            return col_name[:end_index]
        else:
            return col_name

    def _format_settings(self, settings):
        """
        Read in the settings sheet and format it into a dictionary

        Args: settings (ndarray): the raw data from the settings sheet 
        Returns: sheet (dictonary): keys are the method names and values are the integer value
        """
        methods = settings[0]
        method_names = settings[1]
        method_names_size = len(method_names)
        sheet = {}
        for i in range(method_names_size):
            sliced_column_name = self._slice_column_name(method_names[i], " (")
            sheet[sliced_column_name] = methods[i]
            
        return sheet

    def _input_reader(self, file_name):
        """
        Read from the input file

        Args: file_name (string): The name of the input file
        Returns: input_data (dictionary): A dictionary whose indices are sheets' names and 
                                                            values are the corresponding sheet data
        """
        data = pd.ExcelFile(file_name)
        sheet_names = data.sheet_names

        input_data = {}
        for sheet_name in sheet_names:
            sheet = {}
            d = pd.read_excel(data, sheet_name, header=0)
            column_values = d.values.transpose()
            column_names = d.columns.values
            self._format_checking_during_reading(len(column_names), sheet_name)
            if sheet_name == "settings":
                sheet = self._format_settings(column_values)
            else:
                if "Unnamed:" not in column_names[0]:
                    num_cols = len(column_names)
                    for i in range(num_cols):
                        sliced_column_name = self._slice_column_name(column_names[i], " (")
                        if sliced_column_name in sheet.keys():
                            sheet[sliced_column_name] = np.vstack((sheet[sliced_column_name],column_values[i]))
                        else:   
                            sheet[sliced_column_name] = column_values[i]
                else:
                    sheet[sheet_name] = column_values
            input_data[sheet_name] = sheet

        return input_data
    
    def _format_checking_during_reading(self, col_num, sheet_name):
        """
        Check the format of the data to see if they have desirable shapes
        """ 
        if sheet_name in ("temp_x0_data", "temp_t0_data", "T_L_data",
                        "bed_data1", "cloud_data") and col_num != 2:
            print(sheet_name + " must contain 2 columns of data!")
        elif sheet_name == "dim_data" and col_num != 5:
            print(sheet_name + " must contain 5 columns of data!")
        elif sheet_name == "met_data" and col_num != 10:
            print(sheet_name + " must contain 10 columns of data!")
        elif sheet_name == "shade_data" and col_num != 3:
            print(sheet_name + " must contain 3 columns of data!")
        elif sheet_name in ("site_info", "time_mod", "dist_mod", "sed_type") and col_num != 1: 
            print(sheet_name + " must contain 1 columns of data!")

    def _format_checking_after_reading(self):
        """
        Check the format of the data to see if they have desirable shapes
        """ 
        if not self.output_suppression:
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
        if not isinstance(self.temp, np.ndarray):
            raise TypeError("Temp must be a numpy array representing a matrix.")
        
        if not self.output_suppression:
            print('...Done!')


