"""
Author: Michael Scoleri, Violet Shi
Date: 10-02-23
File: hflux_sens.py
Functionality: Implementation of hflux_sens.m
"""

import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from Python.src.Core.heat_flux import HeatFlux
from Python.src.Plotting.plotting_class import Plotting
import numpy as np
import matplotlib.pyplot as plt


class HfluxSens:
    def __init__(self, root_dir):
        self.plc = Plotting()
        self.root_dir = root_dir
        sys.path.append(root_dir)

    def class_assistant(self):
        """
        Print out class details.
        """
        help(type(self))

    @staticmethod
    def heat_flux_wrapper(data_table):
        """
        Wrapper for calling heat flux.

        Args:
            data_table (Data table class object): Class object containing ndarrays of data.

        Returns:
            Result from heat flux calculations.
        """
        return HeatFlux(data_table).crank_nicolson_method()

    @staticmethod
    def multithreading_call(output_suppression, input_data_list, base_result):
        """
        Implements Multi-threading for calls to hflux.py

        Args:
            input_data_list ([dict]): list of dictionaries to be used as the parameter for hflux.py

        Return:
            results ({ndarray, {ndarrarys}, {ndarrays}})
        """
        results = [base_result]

        # Launching Parallel Tasks - https://docs.python.org/3/library/concurrent.futures.html
        if not output_suppression:
            print()
            print("Beginning Multi-threaded calls to hflux...")

        cpu_count = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=cpu_count // 4) as executor:
            for result, _, _, _ in executor.map(
                HfluxSens.heat_flux_wrapper, input_data_list
            ):
                results.append(result)
            executor.shutdown()

        if not output_suppression:
            print("...closed multithreading executer...")

        return results

    @staticmethod
    def singlethread_call(output_suppression, input_data_list, base_result):
        """
        Calls heat_flux.py in a single thread

        Args:
            input_data_list ([dict]): list of dictionaries to be used as the parameter for hflux.py

        Return:
            results ({ndarray, {ndarrarys}, {ndarrays}})
        """
        if not output_suppression:
            print()
            print("Beginning calls to hflux...")

        results = [base_result]
        for result, _, _, _ in map(HfluxSens.heat_flux_wrapper, input_data_list):
            results.append(result)
            if not output_suppression:
                print("Finished a call!")

        if not output_suppression:
            print()
            print("Sensitivity values computed.")

        return results

    def hflux_sens(
        self,
        data_table,
        dis_high_low,
        t_l_high_low,
        vts_high_low,
        shade_high_low,
    ):
        """
        Creates and returns a dictionary with new high and low values.

        Args:
            data_table (data table class object): Class object containing ndarrays of data.
            dis_high_low (ndarray): Ndarray containing distance high and low data.
            t_l_high_low (ndarray): Ndarray containing time high and low data.
            vts_high_low (ndarray): Ndarray containing vts high and low data.
            shade_high_low (ndarray): Ndarray containg shade high and low data.

        Returns:
            High_low_values ({ndarray}): Dictionary containing high and low values.
        """
        output_suppression = data_table.output_suppression
        if not output_suppression:
            print("Calculating high and low values...")

        data_table.output_suppression = True

        high_low_values = {}
        # Create low and high values for each parameter.
        high_low_values["dis_data_1"] = data_table.discharge.transpose()
        high_low_values["dis_low"] = high_low_values["dis_data_1"] + dis_high_low[0]
        high_low_values["dis_high"] = high_low_values["dis_data_1"] + dis_high_low[1]

        high_low_values["t_l_data_1"] = data_table.t_L
        high_low_values["t_l_low"] = high_low_values["t_l_data_1"] + t_l_high_low[0]
        high_low_values["t_l_high"] = high_low_values["t_l_data_1"] + t_l_high_low[1]

        high_low_values["vts"] = data_table.vts
        high_low_values["vts_low"] = high_low_values["vts"] + vts_high_low[0]
        high_low_values["vts_high"] = high_low_values["vts"] + vts_high_low[1]

        high_low_values["shade_1"] = data_table.shade
        high_low_values["shade_low"] = high_low_values["shade_1"] + shade_high_low[0]
        high_low_values["shade_high"] = high_low_values["shade_1"] + shade_high_low[1]

        # Create hflux-ready arrays from the low and high values
        # Use hstack to concatenate along the 2nd axis.
        # Use newaxis to index arrays.
        # np.hstack - https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy-hstack
        # np.newaxis - https://numpy.org/devdocs/reference/constants.html#numpy.newaxis
        high_low_values["dis_data_0"] = data_table.dist_dis
        high_low_values["dis_data_low"] = np.hstack(
            (high_low_values["dis_data_0"][:, np.newaxis], high_low_values["dis_low"])
        ).transpose()
        high_low_values["dis_data_high"] = np.hstack(
            (high_low_values["dis_data_0"][:, np.newaxis], high_low_values["dis_high"])
        ).transpose()

        # Set t_l_data_low and t_l_data_high values.
        high_low_values["t_l_data_0"] = data_table.dist_T_L
        high_low_values["t_l_data_low"] = np.array(
            [high_low_values["t_l_data_0"], high_low_values["t_l_low"]]
        )
        high_low_values["t_l_data_high"] = np.array(
            [high_low_values["t_l_data_0"], high_low_values["t_l_high"]]
        )

        # Set vts_data_low and vts_data_high values.
        high_low_values["shade_0"] = data_table.dist_shade
        high_low_values["vts_data_low"] = np.array(
            [
                high_low_values["shade_0"],
                high_low_values["shade_1"],
                high_low_values["vts_low"],
            ]
        )
        high_low_values["vts_data_high"] = np.array(
            [
                high_low_values["shade_0"],
                high_low_values["shade_1"],
                high_low_values["vts_high"],
            ]
        )

        # Initalize shade_data_low and shade_data_high with zeros.
        high_low_values["shade_2"] = data_table.vts
        high_low_values["shade_data_low"] = np.array(
            [
                high_low_values["shade_0"],
                high_low_values["shade_low"],
                high_low_values["shade_2"],
            ]
        )
        high_low_values["shade_data_high"] = np.array(
            [
                high_low_values["shade_0"],
                high_low_values["shade_high"],
                high_low_values["shade_2"],
            ]
        )

        if not output_suppression:
            print("...done!\n")
            print("Running HLUX for the base, high, and low cases...")

        # Create multiple copies of data_table for and modifying specifc keys.
        high_low_values["input_data_lowdis"] = data_table.modify_data_table(
            "dis_data", high_low_values["dis_data_low"]
        )
        high_low_values["input_data_highdis"] = data_table.modify_data_table(
            "dis_data", high_low_values["dis_data_high"]
        )
        high_low_values["input_data_lowT_L"] = data_table.modify_data_table(
            "t_l_data", high_low_values["t_l_data_low"]
        )
        high_low_values["input_data_highT_L"] = data_table.modify_data_table(
            "t_l_data", high_low_values["t_l_data_high"]
        )
        high_low_values["input_data_lowvts"] = data_table.modify_data_table(
            "shade_data", high_low_values["vts_data_low"]
        )
        high_low_values["input_data_highvts"] = data_table.modify_data_table(
            "shade_data", high_low_values["vts_data_high"]
        )
        high_low_values["input_data_lowshade"] = data_table.modify_data_table(
            "shade_data", high_low_values["shade_data_low"]
        )
        high_low_values["input_data_highshade"] = data_table.modify_data_table(
            "shade_data", high_low_values["shade_data_high"]
        )
        return high_low_values

    def create_new_results(
        self, base_result, high_low_dict, output_suppression, multithread=True
    ):
        """
        Creates and returns a new sensitiviity dictionary

        Args:
            Base_result (ndarray): ndarray containing the temperature modulus data.
            high_low_dict ({ndarray}): Dictionary containing high low values calculated from hflux_sens function.

        Returns:
            Sens ({ndarary}): Dictionary containing sensitivity values.
        """
        new_data_list = [
            high_low_dict["input_data_lowdis"],
            high_low_dict["input_data_highdis"],
            high_low_dict["input_data_lowT_L"],
            high_low_dict["input_data_highT_L"],
            high_low_dict["input_data_lowvts"],
            high_low_dict["input_data_highvts"],
            high_low_dict["input_data_lowshade"],
            high_low_dict["input_data_highshade"],
        ]

        if multithread:
            results = self.multithreading_call(
                output_suppression,
                input_data_list=new_data_list,
                base_result=base_result,
            )
        else:
            results = self.singlethread_call(
                output_suppression,
                input_data_list=new_data_list,
                base_result=base_result,
            )

        temp_mod_base = results[0]
        temp_mod_lowdis = results[1]
        temp_mod_highdis = results[2]
        temp_mod_lowT_L = results[3]
        temp_mod_highT_L = results[4]
        temp_mod_lowvts = results[5]
        temp_mod_highvts = results[6]
        temp_mod_lowshade = results[7]
        temp_mod_highshade = results[8]

        if not output_suppression:
            print("...Done!\n")
            print("Writing output data...\n")

        # Store outputs from hflux to dictionaries.
        base = {"temp": temp_mod_base, "mean": np.mean(temp_mod_base, axis=1)}
        lowdis = {"temp": temp_mod_lowdis, "mean": np.mean(temp_mod_lowdis, axis=1)}
        highdis = {"temp": temp_mod_highdis, "mean": np.mean(temp_mod_highdis, axis=1)}
        lowT_L = {"temp": temp_mod_lowT_L, "mean": np.mean(temp_mod_lowT_L, axis=1)}
        highT_L = {"temp": temp_mod_highT_L, "mean": np.mean(temp_mod_highT_L, axis=1)}
        lowvts = {"temp": temp_mod_lowvts, "mean": np.mean(temp_mod_lowvts, axis=1)}
        highvts = {"temp": temp_mod_highvts, "mean": np.mean(temp_mod_highvts, axis=1)}
        lowshade = {
            "temp": temp_mod_lowshade,
            "mean": np.mean(temp_mod_lowshade, axis=1),
        }
        highshade = {
            "temp": temp_mod_highshade,
            "mean": np.mean(temp_mod_highshade, axis=1),
        }

        # Store structures in dictionary
        sens = {
            "dis_l": high_low_dict["dis_data_low"],
            "dis_h": high_low_dict["dis_data_high"],
            "TL_l": high_low_dict["t_l_data_low"],
            "TL_h": high_low_dict["t_l_data_high"],
            "vts_l": high_low_dict["vts_low"],
            "vts_h": high_low_dict["vts_high"],
            "sh_l": high_low_dict["shade_data_low"],
            "sh_h": high_low_dict["shade_data_high"],
            "base": base,
            "lowdis": lowdis,
            "highdis": highdis,
            "lowT_L": lowT_L,
            "highT_L": highT_L,
            "lowvts": lowvts,
            "highvts": highvts,
            "lowshade": lowshade,
            "highshade": highshade,
        }
        return sens

    def calculate_change(self, sens):
        change = np.array(
            [
                [
                    np.mean(sens["lowdis"]["temp"]) - np.mean(sens["base"]["temp"]),
                    np.mean(sens["highdis"]["temp"]) - np.mean(sens["base"]["temp"]),
                ],
                [
                    np.mean(sens["lowT_L"]["temp"]) - np.mean(sens["base"]["temp"]),
                    np.mean(sens["highT_L"]["temp"]) - np.mean(sens["base"]["temp"]),
                ],
                [
                    np.mean(sens["lowvts"]["temp"]) - np.mean(sens["base"]["temp"]),
                    np.mean(sens["highvts"]["temp"]) - np.mean(sens["base"]["temp"]),
                ],
                [
                    np.mean(sens["lowshade"]["temp"]) - np.mean(sens["base"]["temp"]),
                    np.mean(sens["highshade"]["temp"]) - np.mean(sens["base"]["temp"]),
                ],
            ]
        )
        return change

    # Reshape used to align plotting structures - https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    def make_sens_plots(self, data_table, sens, return_graphs=False):
        """
        Creates and saves plots for sensitiity calculations

        Args:
            Data_table (Data table class object): Class object containing ndarrays of data.
            Sens ({ndarrays}): Sensitivity Dictionary calculted from create_new_results function.

        Returns:
            None
        """
        fig, ax = plt.subplots(2, 2)

        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=sens["lowdis"]["mean"],
            base_y=sens["base"]["mean"],
            high_y=sens["highdis"]["mean"],
            title="Discharge",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[0, 0],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=sens["lowT_L"]["mean"],
            base_y=sens["base"]["mean"],
            high_y=sens["highT_L"]["mean"],
            title="Groundwater Temperature",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[0, 1],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=sens["lowvts"]["mean"],
            base_y=sens["base"]["mean"],
            high_y=sens["highvts"]["mean"],
            title="View to Sky Coefficient",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[1, 0],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=sens["lowshade"]["mean"],
            base_y=sens["base"]["mean"],
            high_y=sens["highshade"]["mean"],
            title="Shade",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[1, 1],
        )
        plt.tight_layout()

        change = self.calculate_change(sens)
        fig2 = self.plc.make_bar_charts(change)

        self.plc.save_plots(fig, fig2, path="hflux_sens")

        if return_graphs:
            return fig, fig2
