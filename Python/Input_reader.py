import pandas as pd
import numpy as np

def readFromFile(filename):
    data = pd.ExcelFile(filename)
    sheet_names = data.sheet_names
    sheet_num = len(sheet_names)

    input_data = {}
    for i in range(0, sheet_num):
        d = pd.read_excel(data, sheet_names[i]).values.transpose()
        formatChecking(len(d), sheet_names[i])
        input_data[sheet_names[i]] = d

    return input_data

def formatChecking(colNum, sheet_name):
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

