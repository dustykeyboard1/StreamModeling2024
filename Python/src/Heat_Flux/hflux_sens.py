"""
Author: Michael Scoleri, Violet Shi
Date: 10-02-23
File: hflux_sens.py
Functionality: Implementation of hflux_sens.m
"""

import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor
from Python.src.Core.heat_flux import HeatFlux
from Python.src.Plotting.plotting_class import Plotting
import numpy as np
import matplotlib.pyplot as plt


class HfluxSens:
    def __init__(self, root_dir):
        self.plc = Plotting()
        self.root_dir = root_dir
        sys.path.append(root_dir)

    @staticmethod
    def heat_flux_wrapper(data_table):
        return HeatFlux(data_table).crank_nicolson_method()

    @staticmethod
    def multithreading_call(input_data_list, base_result):
        """
        Implements Multi-threading for calls to hflux.py

        Args:
            input_data_list ([dict]): list of dictionaries to be used as the parameter for hflux.py

        Return:
            results ({ndarray, {ndarrarys}, {ndarrays}})
        """
        results = [base_result]

        # Launching Parallel Tasks - https://docs.python.org/3/library/concurrent.futures.html
        print()
        print("Beginning Multi-threaded calls to hflux...")
        with ProcessPoolExecutor(max_workers=7) as executor:
            for result, _, _, _ in executor.map(
                HfluxSens.heat_flux_wrapper, input_data_list
            ):
                results.append(result)
            executor.shutdown()
        print()
        print("Closed multithreading executer.")
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
        Implementation of hflux_sens.m
        Parameters: data_table,dis_high_low,t_l_high_low,vts_high_low,shade_high_low
        Returns: Sens - dictionary
        """

        print("Calculating high and low values...")

        data_table.unattend = True

        # Create low and high values for each parameter.
        # dis_data_1 = data_table["dis_data"][1:].transpose()
        dis_data_1 = data_table.discharge.transpose()
        dis_low = dis_data_1 + dis_high_low[0]
        dis_high = dis_data_1 + dis_high_low[1]

        # t_l_data_1 = data_table["T_L_data"][1]
        t_l_data_1 = data_table.t_L
        t_l_low = t_l_data_1 + t_l_high_low[0]
        t_l_high = t_l_data_1 + t_l_high_low[1]

        vts = data_table.vts
        self.vts_low = vts + vts_high_low[0]
        self.vts_high = vts + vts_high_low[1]

        shade_1 = data_table.shade
        shade_low = shade_1 + shade_high_low[0]
        shade_high = shade_1 + shade_high_low[1]

        # Create hflux-ready arrays from the low and high values
        # Use hstack to concatenate along the 2nd axis.
        # Use newaxis to index arrays.
        # np.hstack - https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy-hstack
        # np.newaxis - https://numpy.org/devdocs/reference/constants.html#numpy.newaxis
        dis_data_0 = data_table.dist_dis
        self.dis_data_low = np.hstack((dis_data_0[:, np.newaxis], dis_low)).transpose()
        self.dis_data_high = np.hstack(
            (dis_data_0[:, np.newaxis], dis_high)
        ).transpose()

        # Set t_l_data_low and t_l_data_high values.
        t_l_data_0 = data_table.dist_T_L
        self.t_l_data_low = np.array([t_l_data_0, t_l_low])
        self.t_l_data_high = np.array([t_l_data_0, t_l_high])

        # Set vts_data_low and vts_data_high values.
        shade_0 = data_table.dist_shade
        vts_data_low = np.array([shade_0, shade_1, self.vts_low])
        vts_data_high = np.array([shade_0, shade_1, self.vts_high])

        # Initalize shade_data_low and shade_data_high with zeros.
        shade_2 = data_table.vts
        self.shade_data_low = np.array([shade_0, shade_low, shade_2])
        self.shade_data_high = np.array([shade_0, shade_high, shade_2])

        # Create multiple copies of data_table for and modifying specifc keys.
        print("...Done!")
        print("     ")
        print("Running HLUX for the base, high, and low cases.")

        self.input_data_lowdis = data_table.modify_data_table(
            "dis_data", self.dis_data_low
        )
        self.input_data_highdis = data_table.modify_data_table(
            "dis_data", self.dis_data_high
        )
        self.input_data_lowT_L = data_table.modify_data_table(
            "t_l_data", self.t_l_data_low
        )
        self.input_data_highT_L = data_table.modify_data_table(
            "t_l_data", self.t_l_data_high
        )
        self.input_data_lowvts = data_table.modify_data_table(
            "shade_data", vts_data_low
        )
        self.input_data_highvts = data_table.modify_data_table(
            "shade_data", vts_data_high
        )
        self.input_data_lowshade = data_table.modify_data_table(
            "shade_data", self.shade_data_low
        )
        self.input_data_highshade = data_table.modify_data_table(
            "shade_data", self.shade_data_high
        )

        # Run hlux.m for middle (base) values, then for high and low values of
        # each parameter with other parameters kept at base values

    def create_new_results(self, base_result):
        new_data_list = [
            self.input_data_lowdis,
            self.input_data_highdis,
            self.input_data_lowT_L,
            self.input_data_highT_L,
            self.input_data_lowvts,
            self.input_data_highvts,
            self.input_data_lowshade,
            self.input_data_highshade,
        ]

        results = self.multithreading_call(
            input_data_list=new_data_list, base_result=base_result
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

        print("...Done!")
        print("     ")
        print("Writing output data.")

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
        self.sens = {
            "dis_l": self.dis_data_low,
            "dis_h": self.dis_data_high,
            "TL_l": self.t_l_data_low,
            "TL_h": self.t_l_data_high,
            "vts_l": self.vts_low,
            "vts_h": self.vts_high,
            "sh_l": self.shade_data_low,
            "sh_h": self.shade_data_high,
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
        return self.sens

    def calculate_change(self):
        self.change = np.array(
            [
                [
                    np.mean(self.sens["lowdis"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                    np.mean(self.sens["highdis"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                ],
                [
                    np.mean(self.sens["lowT_L"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                    np.mean(self.sens["highT_L"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                ],
                [
                    np.mean(self.sens["lowvts"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                    np.mean(self.sens["highvts"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                ],
                [
                    np.mean(self.sens["lowshade"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                    np.mean(self.sens["highshade"]["temp"])
                    - np.mean(self.sens["base"]["temp"]),
                ],
            ]
        )

    ####################################################################################################
    # Make sensitivity plots.
    # Following all axes, line, label and function parameters from MATLAB code.
    # Reshape used to align plotting structures - https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    def make_sens_plots(self, data_table):
        fig, ax = plt.subplots(2, 2)

        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=self.sens["lowdis"]["mean"],
            base_y=self.sens["base"]["mean"],
            high_y=self.sens["highdis"]["mean"],
            title="Discharge",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[0, 0],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=self.sens["lowT_L"]["mean"],
            base_y=self.sens["base"]["mean"],
            high_y=self.sens["highT_L"]["mean"],
            title="Groundwater Temperature",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[0, 1],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=self.sens["lowvts"]["mean"],
            base_y=self.sens["base"]["mean"],
            high_y=self.sens["highvts"]["mean"],
            title="View to Sky Coefficient",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[1, 0],
        )
        self.plc.make_three_line_plot(
            x=data_table.dist_mod,
            low_y=self.sens["lowshade"]["mean"],
            base_y=self.sens["base"]["mean"],
            high_y=self.sens["highshade"]["mean"],
            title="Shade",
            xlabel="Distance Downstream(m)",
            ylabel="Temperature (°C)",
            ax=ax[1, 1],
        )
        plt.tight_layout()

        self.calculate_change()
        fig2 = self.plc.make_bar_charts(self.change)

        self.plc.save_plots(fig, fig2, path="hflux_sens")
