import numpy as np
from input_reader import *
from hflux_bed_sed import *

# read from excel sheet
filename = os.getcwd() + "/example_data_formatted.xlsx"
input_data = readFromFile(filename)

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
discharge = np.concatenate((input_data["dis_data"][1], input_data["dis_data"][2]), axis = 0)

time_dis = input_data["time_dis"][0]

dist_T_L = input_data["T_L_data"][0]
T_L = input_data["T_L_data"][1]

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
bed_temp = np.array((input_data["bed_data2"][0, 1:], input_data["bed_data2"][1, 1:]))

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
print(sed)