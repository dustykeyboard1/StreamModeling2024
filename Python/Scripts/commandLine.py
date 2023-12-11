"""
Author: Violet Shi, Michael Scoleri
Date: 12-02-23
File: commandLine.py
Functionality: Execute Hflux from command line prompts
"""
import sys
import os
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from Python.src.Core.heat_flux import HeatFlux
from Python.src.Utilities.data_table_class import DataTable
from Python.src.Plotting.hflux_errors_plotting import create_hflux_errors_plots


def run_heat_flux(input_path, output_path, make_plots, plot_folder_path):
    """
    Run hflux calculations on a given input file.
    Args: input_path (string): the path of the input file
          output_path (string): the path to store output files in
          make_plots (bool): true if plots are expected, false if plots are ignored
          plot_folder_path (string): the path of the folder to store plots in if applicable
    Return: None
    """
    data_table = DataTable(input_path)
    heat_flux = HeatFlux(data_table)
    temp_mod, matrix_data, node_data, flux_data = heat_flux.crank_nicolson_method()
    temp_dt = heat_flux.calculate_temp_dt(temp_mod)
    temp = data_table.temp.transpose()
    rel_err = heat_flux.calculate_percent_relative_error(temp, temp_dt)

    # examples calls to the error calculation methods
    # rmse = heat_flux.calculate_root_mean_square_error(temp, temp_dt)
    # nrmse = heat_flux.calculate_normalized_root_mean_square_error(rmse, temp)
    # me = heat_flux.calculate_mean_residual_error(temp, temp_dt)
    # mae = heat_flux.calculate_mean_absolute_residual_error(temp, temp_dt)
    # mse = heat_flux.calculate_mean_square_error(temp, temp_dt)
    
    dist_temp = data_table.dist_temp
    dist_mod = data_table.dist_mod
    time_temp = data_table.time_temp
    time_mod = data_table.time_mod

    np.savetxt(f"{output_path}/temp_mod.csv", temp_mod, delimiter=",")
    np.savetxt(f"{output_path}/temp.csv", data_table.temp, delimiter=",")
    np.savetxt(f"{output_path}/rel_err.csv", rel_err, delimiter=",")
    np.savetxt(f"{output_path}/heatflux_data.csv", flux_data["heatflux"], delimiter=",")
    np.savetxt(
        f"{output_path}/solarflux_data.csv", flux_data["solarflux"], delimiter=","
    )
    np.savetxt(
        f"{output_path}/solar_refl_data.csv", flux_data["solar_refl"], delimiter=","
    )
    np.savetxt(f"{output_path}/long_data.csv", flux_data["long"], delimiter=",")
    np.savetxt(f"{output_path}/atmflux_data.csv", flux_data["atmflux"], delimiter=",")
    np.savetxt(f"{output_path}/landflux_data.csv", flux_data["landflux"], delimiter=",")
    np.savetxt(f"{output_path}/backrad_data.csv", flux_data["backrad"], delimiter=",")
    np.savetxt(f"{output_path}/evap_data.csv", flux_data["evap"], delimiter=",")
    np.savetxt(f"{output_path}/sensible_data.csv", flux_data["sensible"], delimiter=",")
    np.savetxt(
        f"{output_path}/conduction_data.csv", flux_data["conduction"], delimiter=","
    )

    if make_plots:
        heat_flux.create_hlux_plots(temp_mod, flux_data, sub_directory_path=plot_folder_path, return_graphs=False)
        create_hflux_errors_plots(
            (temp - temp_dt),
            dist_temp,
            temp,
            temp_mod,
            dist_mod,
            time_temp,
            time_mod,
            plot_path=plot_folder_path,
            return_graphs=False
        )


def commandLine_execution():
    """
    Execute the Hflux function with given data.
    Args: None
    Return: None
    """
    # inform the restriction of the data location
    print(
        """\nBefore entering the name of the input file/folder,\nplease make sure this file/folder is in the """
        + "\x1B[4m"
        + "\x1B[1m"
        + """Data folder"""
        + "\x1B[0m"
        + """ of the program!\n"""
    )

    # store name of the input data
    input_name = input("Enter the name of the input file/folder: ")
    input_path = os.path.join(os.getcwd(), "Data", input_name)

    # make sure the input file/folder the user entered exists
    while not os.path.exists(input_path):
        print("Input file/folder" + "\x1B[1m" + " doesn't exist" + "\x1B[0m" + "!\n")
        input_name = input(
            "Enter the"
            + "\x1B[1m"
            + " correct "
            + "\x1B[0m"
            + "name of the input file/folder: "
        )
        input_path = os.path.join(os.getcwd(), "Data", input_name)

    # store the name of the output folder
    output_folder = input("\nEnter the name of the folder to save result data in: ")
    output_path = os.path.join(os.getcwd(), "Results", output_folder)
    os.mkdir(output_path)

    # store whether the user wants plots and if so prepare to store plots
    plot_folder_path = ""
    make_plots = bool(
        int(input("\nEnter 1 to save result plots or 0 to ignore plots: "))
    )
    print()
    if make_plots:
        plot_folder_name = input("Name of the folder to save the plots in: ")
        print()

    # run hflux accordingly
    if os.path.isdir(input_path):
        # store multiple sets of output in their own folders
        file_list = os.listdir(input_path)
        for file in file_list:
            print("\x1B[1m" + "Running " + file + "...\n" + "\x1B[0m")
            file_path = os.path.join(input_path, file)
            file_output_path = os.path.join(output_path, file.rsplit(".")[0])
            os.mkdir(file_output_path)

            file_plot_folder_path = ""
            if make_plots:
                file_plot_folder_path = os.path.join(file_output_path, plot_folder_name)
                os.mkdir(file_plot_folder_path)

            run_heat_flux(
                file_path, file_output_path, make_plots, file_plot_folder_path
            )

            print("\x1B[1m" + "Finished " + file + "!\n" + "\x1B[0m")
    else:
        if make_plots:
            plot_folder_path = os.path.join(
                os.getcwd(), "Results", output_folder, plot_folder_name
            )
            os.mkdir(plot_folder_path)

        run_heat_flux(input_path, output_path, make_plots, plot_folder_path)

    print(
        "See results data/plots in the"
        + "\x1B[1m"
        + " Results "
        + "\x1B[0m"
        + "folder.\n"
    )


if __name__ == "__main__":
    commandLine_execution()
