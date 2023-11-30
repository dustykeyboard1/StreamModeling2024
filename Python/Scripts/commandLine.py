"""
Author: Michael Scoleri
Date: 11-30-23
File: commandLine.py
Functionality: Execute Hflux from commandline
"""
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from Python.src.Core.heat_flux import HeatFlux
from Python.src.Utilities.data_table_class import DataTable


def commandLine_execution():
    """
    Execute the Hflux function with commands from command line.
    Args: None
    Return: None
    """
    is_file = input(
        "Is path a directory, or file? - Enter 0 for Directory, or 1 for File: \n"
    )
    make_plots = input(
        "Do you want to make and save plots for flux calculations? Enter 0 for no, or 1 for yes: \n"
    )
    file_path = input("Please enter the entire file: \n")
    if make_plots:
        save_path = input(
            "Please enter the name of the sub directory to save the plots."
        )

    if is_file:
        data_table = DataTable(file_path)
        heat_flux = HeatFlux(data_table)
        temp_mod, matrix_data, node_data, flux_data = heat_flux.crank_nicolson_method()
        temp_dt = heat_flux.calculate_temp_dt(temp_mod)
        temp = data_table.temp.transpose()
        rel_err = heat_flux.calculate_percent_relative_error(temp, temp_dt)
        me = heat_flux.calculate_mean_residual_error(temp, temp_dt)
        mae = heat_flux.calculate_mean_absolute_residual_error(temp, temp_dt)
        mse = heat_flux.calculate_mean_square_error(temp, temp_dt)
        rmse = heat_flux.calculate_root_mean_square_error(temp, temp_dt)
        nrmse = heat_flux.calculate_normalized_root_mean_square_error(rmse, temp)

        dist_temp = data_table.dist_temp
        dist_mod = data_table.dist_mod
        time_temp = data_table.time_temp
        time_mod = data_table.time_mod
        if make_plots:
            heat_flux.create_hlux_plots(temp_mod, flux_data, save_path)

    else:
        for filename in os.listdir(file_path):
            data_table = DataTable(filename)
            heat_flux = HeatFlux(data_table)
            (
                temp_mod,
                matrix_data,
                node_data,
                flux_data,
            ) = heat_flux.crank_nicolson_method()
            temp_dt = heat_flux.calculate_temp_dt(temp_mod)
            temp = data_table.temp.transpose()
            rel_err = heat_flux.calculate_percent_relative_error(temp, temp_dt)
            me = heat_flux.calculate_mean_residual_error(temp, temp_dt)
            mae = heat_flux.calculate_mean_absolute_residual_error(temp, temp_dt)
            mse = heat_flux.calculate_mean_square_error(temp, temp_dt)
            rmse = heat_flux.calculate_root_mean_square_error(temp, temp_dt)
            nrmse = heat_flux.calculate_normalized_root_mean_square_error(rmse, temp)

            dist_temp = data_table.dist_temp
            dist_mod = data_table.dist_mod
            time_temp = data_table.time_temp
            time_mod = data_table.time_mod
            if make_plots:
                heat_flux.create_hlux_plots(temp_mod, flux_data, (save_path + filename))


if __name__ == "__main__":
    commandLine_execution()
