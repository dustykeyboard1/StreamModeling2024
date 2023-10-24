import pandas as pd
import numpy as np
import os

class DataTable:
    def __init__(self, file_name):
        input_data = self._input_reader(file_name)
        self.unattend = bool(input_data["settings"][0][4])
        # initialize variables
        self.method = input_data["settings"][0][0]
        if not self.unattend:
            print('Assigning variable names...')
        self.settings = input_data["settings"]

        self.time_mod = input_data["time_mod"][0]
        self.dist_mod = input_data["dist_mod"][0]

        self.time_temp = input_data["temp_x0_data"][0]
        self.temp_x0 = input_data["temp_x0_data"][1]

        self.dist_temp = input_data["temp_t0_data"][0]
        self.temp_t0 = input_data["temp_t0_data"][1]

        self.dist_stdim = input_data["dim_data"][0]
        # area = input_data["dim_data"][1]  p.s. commented out in the original Matlab file
        self.width = input_data["dim_data"][2]
        self.depth = input_data["dim_data"][3]
        self.discharge_stdim = input_data["dim_data"][4]

        self.dist_dis = input_data["dis_data"][0]
        self.discharge = input_data["dis_data"][1:]
        self.time_dis = input_data["time_dis"][0]

        self.dist_T_L = input_data["T_L_data"][0]
        self.t_L = input_data["T_L_data"][1]

        self.year = input_data["met_data"][0]
        self.month = input_data["met_data"][1]
        self.day = input_data["met_data"][2]
        self.hour = input_data["met_data"][3]
        self.minute = input_data["met_data"][4]
        self.time_met = input_data["met_data"][5]
        self.solar_rad_in = input_data["met_data"][6]
        self.air_temp_in = input_data["met_data"][7]
        self.rel_hum_in = input_data["met_data"][8]
        self.wind_speed_in = input_data["met_data"][9]

        self.dist_bed = input_data["bed_data1"][0]
        self.depth_of_meas = input_data["bed_data1"][1]

        self.time_bed = np.array([input_data["bed_data2"][0, 0], input_data["bed_data2"][1, 0]])
        self.bed_temp = input_data["bed_data2"][0:, 1:]

        self.sed_type = input_data["sed_type"][0]

        self.dist_shade = input_data["shade_data"][0]
        self.shade = input_data["shade_data"][1]
        self.vts = input_data["shade_data"][2]

        self.time_cloud = input_data["cloud_data"][0]
        self.c_in = input_data["cloud_data"][1]

        self.lat = input_data["site_info"][0, 0]
        self.lon = input_data["site_info"][0, 1]
        self.t_zone = input_data["site_info"][0, 2]
        self.z = input_data["site_info"][0, 3]

        self.temp = input_data['temp']
        self.dis_data = input_data["dis_data"]
        self.t_l_data = input_data["T_L_data"]
        self.shade_data = input_data["shade_data"]

        self._format_checking_after_reading()



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

    def get_input_data(self, file_name):
        return self._input_reader(file_name)


    


    # filename = os.getcwd() + "\\Python\\src\\Utilities" + "\\example_data.xlsx"
    # readFromFile(filename)

