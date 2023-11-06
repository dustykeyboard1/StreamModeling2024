"""
Author: Violet Shi, Michael Scoleri
File: heat_flux.py
Date: 11-05-2023
Functionality: Construct heatflux class to implement Crank Nicolson method and 
                                                     the second order Runge-Kutta method
"""
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, dia_matrix
import scipy.sparse.linalg as linalg

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from src.Utilities.interpolation import interpolation
from src.Heat_Flux.heatflux_calculations import HeatFluxCalculations
from src.Plotting.plotting_class import Plotting


class HeatFlux:
    # minute to second conversion to prevent magic number in the code
    MINUTE_TO_SECOND_CONVERSION = 60

    def __init__(self, data_table):
        """
        Initialize a HeatFlux instance

        Args: data_table (DataTable): data table that stores data from the input file
        """
        self.data_table = data_table
        self.hflux_calculations = HeatFluxCalculations()
        self.plc = Plotting()

    def _calculate_hflux_bed_sed(self):
        """
        Calculate an array of integers that represents the type of sediment of a size that
        corresponds with the model nodes that is used by hflux_flux.m.

        Args: None
        Returns: hflux_bed_sed (ndarray): an array of integers that represents the type of sediment of a size that
                                          corresponds with the model nodes that is used by hflux_flux.m.
        """
        return self.hflux_calculations.hflux_bed_sed(
            self.data_table.sed_type, self.data_table.dist_bed, self.data_table.dist_mod
        )

    def _calculate_sol_refl(self):
        """
        Calculates the reflected shortwave radiation from the stream surface
        using the solar calculations and Fresnel's reflectance.

        Args: None
        Returns: sol_refl (ndarrays): the reflected shortwave radiation from the stream surface
                                      using the solar calculations and Fresnel's reflectance
        """
        return self.hflux_calculations.hflux_shortwave_refl(
            self.data_table.year,
            self.data_table.month,
            self.data_table.day,
            self.data_table.hour,
            self.data_table.minute,
            self.data_table.latitude,
            self.data_table.longitude,
            self.data_table.t_zone,
            self.data_table.time_met,
            self.data_table.time_mod,
        )

    def _interpolate_bed_temp_m(self):
        """
        Interpolate the temperature in the streambed longitudinally so that there are values at every node.

        Args: None
        Returns: bed_temp_m (ndarray): the interpolated the temperature in the streambed
        """
        time_bed_length = len(self.data_table.time_bed)
        bed_temp_m = [0] * time_bed_length
        for i in range(time_bed_length):
            bed_temp_m[i] = interpolation(
                self.data_table.dist_bed,
                self.data_table.bed_temp[i],
                self.data_table.dist_mod,
            )
        bed_temp_m = np.array(bed_temp_m).transpose()

        return bed_temp_m

    def _interpolate_bed_temp_dt(self, r):
        """
        Interpolate the temperature in the streambed through time so that there are values at every time step

        Args: r (int): the length of model distances along the reach where temperature will be calculated(dist_mod)
        Returns: bed_temp_dt (ndarray): the interpolated temperature in the streambed through time
        """
        bed_temp_m = self._interpolate_bed_temp_m()
        bed_temp_dt = [0] * r
        for i in range(r):
            bed_temp_dt[i] = interpolation(
                self.data_table.time_bed, bed_temp_m[i], self.data_table.time_mod
            )
        bed_temp_dt = np.array(bed_temp_dt)

        return bed_temp_dt

    def _interpolate_width_and_depth(self):
        """
        Solve for width of stream through time and distance (assume constant theta and n_s) and
        depth through time and distance (assume constant theta and n_s).

        Args: None
        Returns: width_m (ndarray): width of stream through time and distance
                 depth_m (ndarray): depth through time and distance
        """
        width_m = interpolation(
            self.data_table.dist_stdim, self.data_table.width, self.data_table.dist_mod
        )
        depth_m = interpolation(
            self.data_table.dist_stdim, self.data_table.depth, self.data_table.dist_mod
        )

        return (
            np.arctan((0.5 * width_m) / depth_m),
            interpolation(
                self.data_table.dist_stdim,
                self.data_table.discharge_stdim,
                self.data_table.dist_mod,
            ),
            depth_m,
        )

    def _interpolate_data(self, r):
        """
        Interpolate all data given through time so that there are values at every time step and
        interpolate all input data given longitudinally so that there are values at every node.

        Args: r (int): the length of model distances along the reach where temperature will be calculated(dist_mod)
        Returns:
                t_l_m (ndarray): interpolated temperature of lateral groundwater discharge at each dist
                temp_t0_m (ndarray): interpolated observed temperatures at each dist at time=0
                depth_of_meas_m (ndarray): interpolated depth at which bed temperature measurements were made
                shade_m (ndarray): interpolated values for shading
                vts_m (ndarray): interpolated values for the view to sky coefficient
                solar_rad_mat (ndarray): interpolated total incoming solar radiation data at each time_met
                air_temp_mat (ndarray): interpolated air temperature data at each time_met
                rel_hum_mat (ndarray): interpolated relative humidity data at each time_met
                wind_speed_mat (ndarray): interpolated wind speed data at each time_met
                cl (ndarray): interpolated values for cloud cover
                temp_x0_dt (ndarray): interpolated upstream boundary condition at each time_temp
        """
        t_l_m = interpolation(
            self.data_table.dist_T_L, self.data_table.t_L, self.data_table.dist_mod
        )
        temp_t0_m = interpolation(
            self.data_table.dist_temp, self.data_table.temp_t0, self.data_table.dist_mod
        )
        depth_of_meas_m = interpolation(
            self.data_table.dist_bed,
            self.data_table.depth_of_meas,
            self.data_table.dist_mod,
        )
        shade_m = interpolation(
            self.data_table.dist_shade, self.data_table.shade, self.data_table.dist_mod
        )
        vts_m = interpolation(
            self.data_table.dist_shade, self.data_table.vts, self.data_table.dist_mod
        )

        solar_rad_dt = interpolation(
            self.data_table.time_met,
            self.data_table.solar_rad_in,
            self.data_table.time_mod,
        )
        air_temp_dt = interpolation(
            self.data_table.time_met,
            self.data_table.air_temp_in,
            self.data_table.time_mod,
        )
        rel_hum_dt = interpolation(
            self.data_table.time_met,
            self.data_table.rel_hum_in,
            self.data_table.time_mod,
        )
        wind_speed_dt = interpolation(
            self.data_table.time_met,
            self.data_table.wind_speed_in,
            self.data_table.time_mod,
        )
        c_dt = interpolation(
            self.data_table.time_cloud, self.data_table.c_in, self.data_table.time_mod
        )
        temp_x0_dt = interpolation(
            self.data_table.time_temp,
            self.data_table.temp_x0,
            self.data_table.time_mod,
            "pchip",
        )

        solar_rad_mat = np.array([solar_rad_dt] * r)
        air_temp_mat = np.array([air_temp_dt] * r)
        rel_hum_mat = np.array([rel_hum_dt] * r)
        wind_speed_mat = np.array([wind_speed_dt] * r)
        cl = np.array([c_dt] * r)

        return (
            t_l_m,
            temp_t0_m,
            depth_of_meas_m,
            shade_m,
            vts_m,
            solar_rad_mat,
            air_temp_mat,
            rel_hum_mat,
            wind_speed_mat,
            cl,
            temp_x0_dt,
        )

    def _interpolate_discharge_m(self):
        """
        Interpolate discharge through time so that there are values at every time step and
        interpolate discharge given longitudinally so that there are values at every node.

        Args: None
        Returns: interpolated discharge
        """
        # Need to transpose discharge_m to make sure it has the same shape
        # As discharge_m in Matlab.
        discharge_m = interpolation(
            self.data_table.dist_dis,
            self.data_table.discharge,
            self.data_table.dist_mod,
        ).transpose()
        discharge_m = interpolation(
            self.data_table.time_dis, discharge_m, self.data_table.time_mod
        )

        return discharge_m

    def _calculate_width_depth_discharge_relationship(self, discharge_m):
        """
        Calculate width-depth-discharge relationship.

        Args: discharge_m (ndarray): interpolated known discharge rates, in m3/s, along the reach at each dist
        Returns: width_m (ndarray): interpolated stream width at each dist in meters through time
                 area_m (ndarray): assumes triangular cross sectional area
                 wetted_perimeter_m (ndarray): interpolated wetted perimeter through time and space for triangular cross section
        """
        theta, dim_q, depth_m = self._interpolate_width_and_depth()

        discharge_m = discharge_m.transpose()

        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)

        n_s = (
            (0.25 ** (2 / 3))
            * (2 ** (5 / 3))
            * (cos_theta ** (2 / 3))
            * (tan_theta ** (5 / 3))
            * (depth_m ** (8 / 3))
        ) / (2 * dim_q)

        depth_m = (
            (2 * n_s * discharge_m)
            / (
                (0.25 ** (2 / 3))
                * (2 ** (5 / 3))
                * (cos_theta ** (2 / 3))
                * (tan_theta ** (5 / 3))
            )
        ) ** (3 / 8)

        width_m = (
            2
            * tan_theta
            * (
                (
                    (2 * n_s * discharge_m)
                    / (
                        (0.25 ** (2 / 3))
                        * (2 ** (5 / 3))
                        * (cos_theta ** (2 / 3))
                        * (tan_theta ** (5 / 3))
                    )
                )
                ** (3 / 8)
            )
        )

        area_m = 0.5 * depth_m * width_m
        area_m = area_m.transpose()

        wetted_perimeter_m = (2 * (depth_m / cos_theta)).transpose()
        width_m = width_m.transpose()
        depth_m = depth_m.transpose()

        return width_m, area_m, wetted_perimeter_m

    def _calculate_reservoir_volumes(self, area_m, r, timesteps):
        """
        Compute volumes of each reservoir (node) in the model.

        Args: area_m (ndarray): assumes triangular cross sectional area
              r (int): the length of model distances along the reach where temperature will be calculated(dist_mod)
              timesteps (int): the length of model times at which temperatures will be computed for heat budget(time_mod)

        Returns: volume (ndarray):
        """
        if not self.data_table.output_suppression:
            print(
                "Computing volumes of nodes, discharge rates and groundwater inflow rates..."
            )
        volume = np.empty((r, timesteps))
        volume[0] = (
            self.data_table.dist_mod[1] - self.data_table.dist_mod[0]
        ) * area_m[0]
        volume[1 : r - 1] = (
            area_m[1 : r - 1].transpose()
            * (
                (
                    (self.data_table.dist_mod[2:] + self.data_table.dist_mod[1 : r - 1])
                    / 2
                )
                - (
                    (
                        self.data_table.dist_mod[1 : r - 1]
                        + self.data_table.dist_mod[: r - 2]
                    )
                    / 2
                )
            )
        ).transpose()
        volume[r - 1] = area_m[r - 1] * (
            self.data_table.dist_mod[r - 1] - self.data_table.dist_mod[r - 2]
        )

        return volume

    def _linearly_calculate_reservoir_edges_discharge_rates(
        self, r, timesteps, discharge_m
    ):
        """
        Compute discharge rates at reservoir edges using linear interpolation (n-values are upstream of node n values).

        Args: r (int): the length of model distances along the reach where temperature will be calculated(dist_mod)
              timesteps (int): the length of model times at which temperatures will be computed for heat budget(time_mod)
              discharge_m (ndarray): interpolated known discharge rates, in m3/s, along the reach at each dist

        Returns: q_half (ndarray): discharge rates at reservoir edges using linear interpolation
                                   (n-values are upstream of node n values)
        """
        q_half = np.empty((r + 1, timesteps))
        q_half[0] = (2 * discharge_m[0] - discharge_m[1] + discharge_m[0]) / 2
        q_half[1:r] = (discharge_m[1:r] + discharge_m[0 : r - 1]) / 2
        q_half[r] = (
            2 * discharge_m[r - 1] - discharge_m[r - 2] + discharge_m[r - 1]
        ) / 2

        return q_half * self.MINUTE_TO_SECOND_CONVERSION

    def _compute_lateral_groundwater_discharge_rates_through_longtudinal_changes(
        self, r, q_half
    ):
        """
        Compute lateral groundwater discharge rates to each node based on longtudinal changes in streamflow.

        Args: r (int): the length of model distances along the reach where temperature will be calculated(dist_mod)
              q_half (ndarray): discharge rates at reservoir edges using linear interpolation
                                (n-values are upstream of node n values)
        Returns: q_l (ndarray): lateral groundwater discharge rates to each node based on longtudinal changes in streamflow
        """
        q_l = q_half[1 : r + 1] - q_half[:r]

        return q_l

    def crank_nicolson_method(self):
        """
        Calculate stream temperature using Crank Nicolson method.

        Args: None

        Returns: temp_mod (ndarray): modeled stream temperatures through space and time (deg C)
                 matrix_data (dictonary): a structure containing the intermediate steps used
                                          to calculate the finite difference solution.
                                          Includes a,b,c,A,o,p,q,g,k,m, and d
                 node_data (dictonary): a structure containing the volume of each node (volume, m3),
                                        the interpolated discharge measurements at each
                                        node (Q, m3/min), and the lateral inflow calculated at each
                                        node (ql, m3/min)
                 flux_data (dictonary): a structure containing the total heat flux (heatflux),
                                        solar radiation (solarflux), atmospheric longwave
                                        radiation (atmflux), land cover radiation (landflux),
                                        back radiation from the stream (backrad), latent
                                        heat flux (evap),stream bed conduction
                                        (conduction), and sensible heat flux (sens), all
                                        in W/m2, at each node and time_step
        """
        output_suppression = self.data_table.output_suppression
        time_mod = self.data_table.time_mod
        timesteps = len(time_mod)

        dist_mod = self.data_table.dist_mod
        r = len(dist_mod)
        dt = max(time_mod) / (timesteps - 1)

        discharge_m = self._interpolate_discharge_m()
        (
            width_m,
            area_m,
            wetted_perimeter_m,
        ) = self._calculate_width_depth_discharge_relationship(discharge_m)
        volume = self._calculate_reservoir_volumes(area_m, r, timesteps)

        q_half_min = self._linearly_calculate_reservoir_edges_discharge_rates(
            r, timesteps, discharge_m
        )
        q_l_min = self._compute_lateral_groundwater_discharge_rates_through_longtudinal_changes(
            r, q_half_min
        )

        (
            t_l_m,
            temp_t0_m,
            depth_of_meas_m,
            shade_m,
            vts_m,
            solar_rad_mat,
            air_temp_mat,
            rel_hum_mat,
            wind_speed_mat,
            cl,
            temp_x0_dt,
        ) = self._interpolate_data(r)

        sed = self._calculate_hflux_bed_sed()
        sol_refl = self._calculate_sol_refl()
        bed_temp_dt = self._interpolate_bed_temp_dt(r)

        ###############################################################
        # Calculate coefficients of the tridiagonal matrix (a, b, c)
        # and set coefficients at the boundaries. Use a, b and c to create the A
        # matrix.  Note that a, b and c are constant in time as long as Q,
        # volume, and q_l are constant with time.

        # Prepare to calculate coefficients of the tridiagonal matrix(a, b, c)
        double_volume = 2 * volume[:, 0]
        quad_volume = 4 * volume[:, 0]

        a = np.empty((r, timesteps))
        a[:, : timesteps - 1] = (
            (-dt * q_half_min[:r, 1:]).transpose() / quad_volume
        ).transpose()
        a[:, timesteps - 1] = (-dt * q_half_min[:r, timesteps - 1]) / quad_volume

        b = np.empty((r, timesteps))
        o = np.empty((r, timesteps))
        p = np.empty((r, timesteps))
        q = np.empty((r, timesteps))
        o[:, : timesteps - 1] = (
            (dt * q_half_min[:r, 1:]).transpose() / quad_volume
        ).transpose()
        p[:, : timesteps - 1] = (
            (dt * q_half_min[1:, 1:]).transpose() / quad_volume
        ).transpose()
        q[:, : timesteps - 1] = (
            (dt * q_l_min[:, : timesteps - 1]).transpose() / double_volume
        ).transpose()
        o[:, timesteps - 1] = (
            (dt * q_half_min[:r, timesteps - 1]).transpose() / quad_volume
        ).transpose()
        p[:, timesteps - 1] = (
            (dt * q_half_min[1:, timesteps - 1]).transpose() / quad_volume
        ).transpose()
        q[:, timesteps - 1] = (
            (dt * q_l_min[:r, timesteps - 1]).transpose() / double_volume
        ).transpose()
        b[:, : timesteps - 1] = (
            1 + o[:, : timesteps - 1] - p[:, : timesteps - 1] + q[:, : timesteps - 1]
        )
        b[:, timesteps - 1] = (
            1 + o[:, timesteps - 1] - p[:, timesteps - 1] + q[:, timesteps - 1]
        )

        c = np.empty((r, timesteps))
        c[:, : timesteps - 1] = (
            (dt * q_half_min[1:, 1:]).transpose() / quad_volume
        ).transpose()
        c[:, timesteps - 1] = (dt * q_half_min[1:, timesteps - 1]) / quad_volume

        a_c = ((-dt * q_half_min[:r]).transpose() / quad_volume).transpose()
        o_c = ((dt * q_half_min[:r]).transpose() / quad_volume).transpose()
        p_c = ((dt * q_half_min[1:, :]).transpose() / quad_volume).transpose()
        q_c = ((dt * q_l_min).transpose() / double_volume).transpose()
        b_c = (1 + o_c - p_c + q_c).transpose()
        c_c = ((dt * q_half_min[1:, :]).transpose() / quad_volume).transpose()

        ###############################################################
        # Calculate right hand side (d).
        # The values for d are temperature-dependent, so they change each time step.
        # Once d is computed, use that d value and the
        # matrix A to solve for the temperature for each time step.
        if not output_suppression:
            print(
                "Computing d-values, heat fluxes and solving for stream temperatures..."
            )

        d = np.empty((r, timesteps))
        temp_mod = np.empty((r, timesteps))
        temp_mod[:, 0] = temp_t0_m
        heat_flux = np.empty((r, timesteps))
        shortwave = np.empty((r, timesteps))
        longwave = np.empty((r, timesteps))
        atm = np.empty((r, timesteps))
        back = np.empty((r, timesteps))
        land = np.empty((r, timesteps))
        latent = np.empty((r, timesteps))
        sensible = np.empty((r, timesteps))
        bed = np.empty((r, timesteps))

        (
            heat_flux[:, 0],
            shortwave[:, 0],
            longwave[:, 0],
            atm[:, 0],
            back[:, 0],
            land[:, 0],
            latent[:, 0],
            sensible[:, 0],
            bed[:, 0],
        ) = self.hflux_calculations.heatflux_calculations(
            self.data_table.settings,
            solar_rad_mat[:, 0],
            air_temp_mat[:, 0],
            rel_hum_mat[:, 0],
            temp_t0_m,
            wind_speed_mat[:, 0],
            self.data_table.z,
            sed,
            bed_temp_dt[:, 0],
            depth_of_meas_m,
            shade_m,
            vts_m,
            cl[:, 0],
            sol_refl[0],
            wetted_perimeter_m[:r, 0],
            width_m[:, 0],
        )

        # Density of water in kg/m^3
        rho_water = 1000
        # Specific heat capacity of water in J /kg deg C
        c_water = 4182

        g = 1 + a_c + c_c - q_c

        k = np.empty((r, timesteps))
        for i in range(r):
            for j in range(timesteps):
                if q_l_min[i, j] < 0:
                    k[i, j] = 0
                else:
                    k[i, j] = (dt * q_l_min[i, j]) / (volume[i, 0])

        m = np.zeros(width_m.shape)
        d = np.zeros(width_m.shape)

        for i in range(timesteps - 1):
            m[:, i] = (
                dt
                * (width_m[:, i] * heat_flux[:, i])
                / ((rho_water * c_water))
                / area_m[:, i]
            )
            d[0, i] = (
                (g[0, i] * temp_mod[0, i])
                + (o_c[0, i] * temp_x0_dt[i])
                - (p_c[0, i] * temp_mod[1, i])
                + (k[0, i] * t_l_m[0])
                + m[0, i]
            ) - (a_c[0, i] * temp_x0_dt[i + 1])
            d[r - 1, i] = (
                (g[r - 1, i] * temp_mod[r - 1, i])
                + (o_c[r - 1, i] * temp_mod[r - 2, i])
                - (p_c[r - 1, i] * temp_mod[r - 1, i])
                + (k[r - 1, i] * t_l_m[r - 1])
                + m[r - 1, i]
            )
            d[1 : r - 1, i] = (
                (g[1 : r - 1, i] * temp_mod[1 : r - 1, i])
                + (o_c[1 : r - 1, i] * temp_mod[0 : r - 2, i])
                - (p_c[1 : r - 1, i] * temp_mod[2:r, i])
                + (k[1 : r - 1, i] * t_l_m[1 : r - 1])
                + m[1 : r - 1, i]
            )

            # Constructing the compressed diagonal matrix
            # CITE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html
            b_row = np.append(b[: r - 1, i], [b[r - 1, i] + c[r - 1, i]])
            a_row = np.append(a[1:r, i], [0])
            c_row = np.append([0], c[: r - 1, i])
            data = np.stack((a_row, b_row, c_row))
            offsets = np.array([-1, 0, 1])
            A = dia_matrix((data, offsets), shape=(r, r))

            # convert A into a csc matrix to efficiently solve for temperature
            # CITE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
            A = csc_matrix(A)
            temp_mod[:, i + 1] = linalg.splu(A).solve(d[:, i])

            (
                heat_flux[:, i + 1],
                shortwave[:, i + 1],
                longwave[:, i + 1],
                atm[:, i + 1],
                back[:, i + 1],
                land[:, i + 1],
                latent[:, i + 1],
                sensible[:, i + 1],
                bed[:, i + 1],
            ) = self.hflux_calculations.heatflux_calculations(
                self.data_table.settings,
                solar_rad_mat[:, i + 1],
                air_temp_mat[:, i + 1],
                rel_hum_mat[:, i + 1],
                temp_mod[:, i + 1],
                wind_speed_mat[:, i + 1],
                self.data_table.z,
                sed,
                bed_temp_dt[:, i + 1],
                depth_of_meas_m,
                shade_m,
                vts_m,
                cl[:, i + 1],
                sol_refl[i + 1],
                wetted_perimeter_m[:, i + 1],
                width_m[:, i + 1],
            )

        # Constructing outputs
        matrix_data = {}
        matrix_data["a"] = a
        matrix_data["b"] = b
        matrix_data["c"] = c
        matrix_data["A"] = A
        matrix_data["o"] = o
        matrix_data["p"] = p
        matrix_data["q"] = q
        matrix_data["g"] = g
        matrix_data["k"] = k
        matrix_data["m"] = m
        matrix_data["d"] = d

        matrix_data["a_c"] = a_c
        matrix_data["b_c"] = b_c
        matrix_data["c_c"] = c_c
        matrix_data["o_c"] = o_c
        matrix_data["p_c"] = p_c
        matrix_data["q_c"] = q_c

        node_data = {}
        node_data["v"] = volume
        node_data["Q"] = q_half_min
        node_data["ql"] = q_l_min
        node_data["width"] = width_m
        node_data["area"] = area_m

        flux_data = {}
        flux_data["heatflux"] = heat_flux / self.MINUTE_TO_SECOND_CONVERSION
        flux_data["solarflux"] = shortwave
        flux_data["solar_refl"] = sol_refl
        flux_data["long"] = longwave
        flux_data["atmflux"] = atm
        flux_data["landflux"] = land
        flux_data["backrad"] = back
        flux_data["evap"] = latent
        flux_data["sensible"] = sensible
        flux_data["conduction"] = bed

        return temp_mod, matrix_data, node_data, flux_data

    def runge_kutta_method():
        """
        (Imcomplete due to lack of data)
        Calculate stream temperature using a second order Runge-Kutta method.
        """
        #     temp_mod = np.zeros((r, timesteps))
        #     temp_mod[:,0] = temp_t0_m
        #     temp_mod[0,:] = temp_x0_dt
        #     t_k1 = np.zeros((r, timesteps))
        #     heat_flux = np.zeros((r, timesteps))

        #     heat_flux = np.empty((r, timesteps))
        #     heat_flux_k1 = np.empty((r, timesteps))
        #     heat_flux_k2 = np.empty((r, timesteps))
        #     shortwave = np.empty((r, timesteps))
        #     longwave = np.empty((r, timesteps))
        #     atm = np.empty((r, timesteps))
        #     back = np.empty((r, timesteps))
        #     land = np.empty((r, timesteps))
        #     latent = np.empty((r, timesteps))
        #     sensible = np.empty((r, timesteps))
        #     bed = np.empty((r, timesteps))
        #     u1 = np.empty((r, timesteps))
        #     v1 = np.empty((r, timesteps))
        #     s1 = np.empty((r, timesteps))
        #     m1 = np.empty((r, timesteps))
        #     k1 = np.empty((r, timesteps))
        #     u2 = np.empty((r, timesteps))
        #     v2 = np.empty((r, timesteps))
        #     s2 = np.empty((r, timesteps))
        #     m2 = np.empty((r, timesteps))
        #     k2 = np.empty((r, timesteps))

        #     heat_flux[:, 0], shortwave[:, 0], longwave[:, 0], atm[:, 0], back[:, 0], land[:, 0], latent[:, 0], sensible[:, 0], bed[:, 0] = hflux_flux(input_data["settings"], solar_rad_mat[:, 0],
        #                                             air_temp_mat[:, 0], rel_hum_mat[:, 0], temp_t0_m,
        #                                             wind_speed_mat[:, 0], z, sed, bed_temp_dt[:, 0],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, 0], sol_refl[0], wetted_perimeter_m[:r, 0], width_m[:, 0])

        #     for i in range(timesteps - 1):
        #         heat_flux_k1[:, i], _, _, _, _, _, _, _, _ = hflux_flux(input_data["settings"], solar_rad_mat[:, i],
        #                                             air_temp_mat[:, i], rel_hum_mat[:, i], temp_mod[:, i],
        #                                             wind_speed_mat[:, i], z, sed, bed_temp_dt[:, i],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, i], sol_refl[i], wetted_perimeter_m[:, i], width_m[:, i])

        #         for j in range(1, r - 1):
        #             u1[j, i] = (q_half_min[j,i] / volume[j, 0]) * (.5 * temp_mod[j - 1, i] - .5 * temp_mod[j, i])
        #             v1[j, i]=(q_half_min[j + 1, i] / volume[j, 0]) * (0.5 * temp_mod[j, i] - 0.5 * temp_mod[j + 1,i])
        #             s1[j, i]=(q_l_min[j, i] / volume[j, 0]) * (t_l_m[j] - temp_mod[j, i])

        #             rho_water = 1000
        #             c_water = 4182
        #             m1[j, i] = (width_m[j,i] * heat_flux_k1[j,i]) / ((rho_water*c_water)) / area_m[j, i]
        #             k1[j, i] = u1[j, i] + v1[j, i] + s1[j, i] + m1[j, i]

        #         u1[r - 1, i] = (q_half_min[r - 1,i] / volume[r - 1, 0]) * (.5 * temp_mod[r - 2, i] - .5 * temp_mod[r - 1, i])
        #         v1[r - 1, i]=(q_half_min[r, i] / volume[r - 1, 0]) * (0.5 * temp_mod[r - 1, i] - 0.5 * temp_mod[r - 1,i])
        #         s1[r - 1, i]=(q_l_min[r - 1, i] / volume[r - 1, 0]) * (t_l_m[r - 1] - temp_mod[r - 1, i])

        #         m1[r - 1, i] = (width_m[r - 1,i] * heat_flux_k1[r - 1,i]) / ((rho_water*c_water)) / area_m[r - 1, i]
        #         k1[r - 1, i] = u1[r - 1, i] + v1[r - 1, i] + s1[r - 1, i] + m1[r - 1, i]

        #         # Calculate temp based on k1
        #         t_k1[0, i] = temp_x0_dt[i]
        #         for j in range(1, r):
        #             t_k1[j, i] = temp_mod[j, i] + (dt * k1[j, i])

        #         heat_flux_k2[:, i], _, _, _, _, _, _, _, _ = hflux_flux(input_data["settings"], solar_rad_mat[:, i],
        #                                             air_temp_mat[:, i], rel_hum_mat[:, i], t_k1[:, i],
        #                                             wind_speed_mat[:, i], z, sed, bed_temp_dt[:, i],
        #                                             depth_of_meas_m, shade_m, vts_m,
        #                                             cl[:, i], sol_refl[i], wetted_perimeter_m[:, i], width_m[:, i])

        #         for j in range(1, r - 1):
        #             u2[j, i] = (q_half_min[j, i + 1] / volume[j, 0]) * (.5 * t_k1[j - 1, i] - .5 * t_k1[j, i])
        #             v2[j, i]=(q_half_min[j + 1, i + 1] / volume[j, 0]) * (0.5 * t_k1[j, i] - 0.5 * t_k1[j + 1,i])
        #             s2[j, i]=(q_l_min[j, i + 1] / volume[j, 0]) * (t_l_m[j] - t_k1[j, i])

        #             m2[j, i] = (width_m[j,i] * heat_flux_k2[j,i]) / ((rho_water*c_water)) / area_m[j, i]
        #             k2[j, i] = u2[j, i] + v2[j, i] + s2[j, i] + m2[j, i]

        #         u2[r - 1, i] = (q_half_min[r - 1, i + 1] / volume[r - 1, 0]) * (.5 * t_k1[r - 2, i] - .5 * t_k1[r - 1, i])
        #         v2[r - 1, i]=(q_half_min[r, i] / volume[r - 1, 0]) * (0.5 * t_k1[r - 1, i] - 0.5 * t_k1[r - 1,i])
        #         s2[r - 1, i]=(q_l_min[r - 1, i] / volume[r - 1, 0]) * (t_l_m[r - 1] - t_k1[r - 1, i])

        #         m2[r - 1, i] = (width_m[r - 1,i] * heat_flux_k2[r - 1,i]) / ((rho_water*c_water)) / area_m[r - 1, i]
        #         k2[r - 1, i] = u2[r - 1, i] + v2[r - 1, i] + s2[r - 1, i] + m2[r - 1, i]

        #         for j in range(1, r):
        #             temp_mod[j, i + 1] = temp_mod[j, i] + (dt * (.5 * k1[j, i]) + .5 * k2[j, i])

        #         heat_flux[:,i+1], shortwave[:,i+1], longwave[:,i + 1], atm[:,i + 1], back[:,i + 1], land[:,i + 1], latent[:,i + 1],sensible[:,i + 1], bed[:,i + 1] = hflux_flux(input_data["settings"],
        #             solar_rad_mat[:,i+1],air_temp_mat[:,i+1],
        #             rel_hum_mat[:,i+1],temp_mod[:,i+1],wind_speed_mat[:,i+1],z,
        #             sed,bed_temp_dt[:,i+1],depth_of_meas_m,
        #             shade_m,vts_m,cl[:,i+1],sol_refl[i + 1], wetted_perimeter_m[:,i+1], width_m[:,i+1])
        #         print(i)
        #     matrix_data["u1"] = u1
        #     matrix_data["v1"] = v1
        #     matrix_data["s1"] = s1
        #     matrix_data["m1"] = m1
        #     matrix_data["k1"] = k1
        #     matrix_data["u2"] = u2
        #     matrix_data["v2"] = v2
        #     matrix_data["s2"] = s2
        #     matrix_data["m2"] = m2
        #     matrix_data["k2"] = k2
        pass

    def calculate_temp_dt(self, temp_mod):
        """
        Resample the modeled temperature data to the same temporal and
        spatial scales as the measured temperature data.

        Args: temp_mod (ndarray): modeled temperature
        Returns: temp_dt(ndarray): Resampled the modeled temperature data that has the same temporal and
                                    spatial scales as the measured temperature data
        """
        time_mod = self.data_table.time_mod
        dist_mod = self.data_table.dist_mod
        dist_temp = self.data_table.dist_temp
        time_temp = self.data_table.time_temp
        result_list = []
        for i in range(len(time_mod)):
            result = interpolation(dist_mod, temp_mod[:, i], dist_temp)
            result_list.append(result)
        temp_dx = np.array(result_list).transpose()

        # Performs linear interpolation using time_mod, temp_dx, time_temp at each time step.
        # Stores in temp_dt.
        result_list = []
        for i in range(len(dist_temp)):
            result = interpolation(time_mod, temp_dx[i, :], time_temp)
            result_list.append(result)
        temp_dt = np.array(result_list)

        return temp_dt

    def calculate_percent_relative_error(self, temp, temp_dt):
        """
        Calculate percent relative error.

        Args: temp (ndarry): measured temperature data
              temp_dt (ndarray): Resampled the modeled temperature data that has the same temporal and
                                 spatial scales as the measured temperature data

        Returns: percent_relative_error (ndarray)
        """
        return ((temp - temp_dt) / temp) * 100

    def calculate_mean_residual_error(self, temp, temp_dt):
        """
        Calculate mean residual error.

        Args: temp (ndarry): measured temperature data
              temp_dt (ndarray): Resampled the modeled temperature data that has the same temporal and
                                 spatial scales as the measured temperature data

        Returns: mean_residual_error (float)
        """
        return np.sum(temp - temp_dt) / np.size(temp)

    def calculate_mean_absolute_residual_error(self, temp, temp_dt):
        """
        Calculate mean absolute residual error.

        Args: temp (ndarry): measured temperature data
              temp_dt (ndarray): Resampled the modeled temperature data that has the same temporal and
                                 spatial scales as the measured temperature data

        Returns: mean_absolute_residual_error (float)
        """
        return np.sum(np.abs(temp - temp_dt)) / np.size(temp)

    def calculate_mean_square_error(self, temp, temp_dt):
        """
        Calculate mean square error.

        Args: temp (ndarry): measured temperature data
              temp_dt (ndarray): Resampled the modeled temperature data that has the same temporal and
                                 spatial scales as the measured temperature data

        Returns: mean_squared_error (float)
        """
        return np.sum((temp - temp_dt) ** 2) / np.size(temp)

    def calculate_root_mean_square_error(self, temp, temp_dt):
        """
        Calculate root mean square error.

        Args: temp (ndarry): measured temperature data
              temp_dt (ndarray): Resampled the modeled temperature data that has the same temporal and
                                 spatial scales as the measured temperature data

        Returns: root_mean_squared_error (float)
        """
        return np.sqrt(np.sum((temp - temp_dt) ** 2) / np.size(temp))

    def calculate_normalized_root_mean_square_error(
        self, root_mean_squared_error, temp
    ):
        """
        Calculate normalized root mean square error.

        Args: root_mean_squared_error (float): the root mean squared error to be normalized
              temp (ndarry): measured temperature data

        Returns: normalized_root_mean_squared_error (float)
        """
        return (root_mean_squared_error / (np.max(temp) - np.min(temp))) * 100

    def create_hlux_plots(self, temp_mod, flux_data):
        """
        Main method for creating and saving heat flux plots.

        Args:
            temp_mod (ndarray): ndarray containing the time component.
            flux_data ({ndarray}): dictionary containing other ndarrays for plotting.

        Returns:
            None.
        """
        hflux_resiudal = self.flux_residual_plot(temp_mod)
        hflux_3d = self.hlfux_3d_plot(temp_mod)
        hflux_subplots = self.make_subplots(flux_data)
        comparison_plot = self.make_comparison_plot(flux_data)
        self.plc.save_plots(
            hflux_resiudal, hflux_3d, hflux_subplots, comparison_plot, path="hflux"
        )

    def flux_residual_plot(self, temp_mod):
        """
        Makes a call to the plotting class and returns a heat map figure.

        Args:
            temp_mod (ndarray): numpy array containing time data.

        Returns:
            fig (matplotlib figure): figure containing heat map.
        """
        plot_title = "Modeled Stream Temperature"
        xlab = "Time (min)"
        ylab = "Distance (m)"
        clab = "Temperature (°C)"
        fig = self.plc.make_residual_plot(
            data=temp_mod,
            xlabel=xlab,
            ylabel=ylab,
            title=plot_title,
            colorbar_label=clab,
            extent=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(self.data_table.dist_mod),
                np.max(self.data_table.dist_mod),
            ],
        )
        return fig

    def hlfux_3d_plot(self, temp_mod):
        """
        Makes a call to the plotting class and returns a 3d figure.

        Args:
            temp_mod (ndarray): numpy array containing time data.

        Returns:
            fig (matplotlib figure): figure containing 3d plot.
        """
        fig = self.plc.make3dplot(
            self.data_table.time_mod,
            self.data_table.dist_mod,
            temp_mod,
            ylabel="Distance (m)",
            xlabel="Time (min)",
            zlabel="Temp (°C)",
            colorbar_label="Temperature (°C)",
            title="Modeled Stream Temperature",
        )
        return fig

    def make_subplots(self, flux_data):
        """
        Creates a figure of 6 subplots.

        Args:
            flux_data ({ndarray}): dictionary containing ndarrys for data to plot.

        Returns:
            fig (matplotlib figure): Figure containing all 6 subplots.
        """
        fig, axs = plt.subplots(3, 2)

        # First Subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["heatflux"], axis=0),
            ylabel="Energy Flux (W/m^2)",
            title="Total Heat Flux",
            marker="k",
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["heatflux"], axis=0)),
                np.max(np.mean(flux_data["heatflux"], axis=0)),
            ],
            xlabel=None,
            ax=axs[0, 0],
        )

        # Second subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["solarflux"], axis=0),
            marker="r",
            title="Shortwave Radiation",
            xlabel=None,
            ylabel=None,
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["solarflux"], axis=0)),
                np.max(np.mean(flux_data["solarflux"], axis=0)),
            ],
            ax=axs[0, 1],
        )

        # Third Subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["long"], axis=0),
            marker="b",
            title="Longwave Radiation",
            ylabel="Energy Flux (W/m^2)",
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["long"], axis=0)),
                np.max(np.mean(flux_data["long"], axis=0)),
            ],
            xlabel=None,
            ax=axs[1, 0],
        )

        # Fourth Subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["evap"], axis=0),
            marker="g",
            title="Latent Heat Flux",
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["evap"], axis=0)),
                np.max(np.mean(flux_data["evap"], axis=0)),
            ],
            ax=axs[1, 1],
            ylabel=None,
            xlabel=None,
        )

        # Fifth Subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["conduction"], axis=0),
            marker="m",
            title="Bed Conduction",
            xlabel="Time (min)",
            ylabel="Energy Flux (W/m^2)",
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["conduction"], axis=0)),
                np.max(np.mean(flux_data["conduction"], axis=0)),
            ],
            ax=axs[2, 0],
        )

        # Sixth Subplot
        self.plc.make_single_plot(
            x=self.data_table.time_mod,
            y=np.mean(flux_data["sensible"], axis=0),
            title="Sensible Heat Flux",
            xlabel="Time (min)",
            ylabel=None,
            axis=[
                np.min(self.data_table.time_mod),
                np.max(self.data_table.time_mod),
                np.min(np.mean(flux_data["sensible"], axis=0)),
                np.max(np.mean(flux_data["sensible"], axis=0)),
            ],
            ax=axs[2, 1],
            marker="y",
        )
        plt.tight_layout()
        return fig

    def make_comparison_plot(self, flux_data):
        """
        Creates a figure with 6 different values on the same plot to compare.

        Args:
            flux_data ({ndarray}): dictionary containing ndarrys for data to plot.

        Returns:
            fig (matplotlib figure): Figure containing all 6 values on a single plot.
        """
        fig = self.plc.heat_flux_comparison(
            x=self.data_table.time_mod,
            y1=np.mean(flux_data["heatflux"], axis=0),
            marker1="k",
            label1="Total Heat Flux",
            y2=np.mean(flux_data["solarflux"], axis=0),
            marker2="r",
            label2="Solar Radiation",
            y3=np.mean(flux_data["long"], axis=0),
            marker3="b",
            label3="Longwave Radiation",
            y4=np.mean(flux_data["evap"], axis=0),
            marker4="g",
            label4="Latent Heat Flux",
            y5=np.mean(flux_data["conduction"], axis=0),
            marker5="c",
            label5="Streambed Conduction",
            y6=np.mean(flux_data["sensible"], axis=0),
            marker6="m",
            label6="Sensible Heat Flux",
            title="Energy Fluxes",
            xlabel="Time (min)",
            ylabel="Energy Flux (W/m^2)",
        )
        return fig
