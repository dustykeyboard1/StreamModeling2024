import pandas as pd
import numpy as np
import os

class DataTable:
    def __init__(self, filename):
        data = pd.ExcelFile(filename)
        sheet_names = data.sheet_names
        sheet_num = len(sheet_names)

        input_data = {}
        for i in range(0, sheet_num):
            d = pd.read_excel(data, sheet_names[i]).values.transpose()
            self.formatChecking(len(d), sheet_names[i])
            input_data[sheet_names[i]] = d

        self.method = input_data["settings"][0][0]
        self.unattend = bool(input_data["settings"][0][4])

        if not self.unattend:
            print('Assigning variable names...')

        # initialize variables
        self.time_mod = input_data["time_mod"][0]
        self.dist_mod = input_data["dist_mod"][0]

        self.time_temp = input_data["temp_x0_data"][0]
        self.temp_x0 = input_data["temp_x0_data"][1]

        self.dist = input_data["temp_t0_data"][0]
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


    def _formatChecking(colNum, sheet_name):
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

    


    # filename = os.getcwd() + "\\Python\\src\\Utilities" + "\\example_data.xlsx"
    # readFromFile(filename)

