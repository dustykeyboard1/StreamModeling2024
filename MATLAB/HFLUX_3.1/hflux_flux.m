function [net, shortwave, longwave, atm, back, land, latent, sensible, bed] = hflux_flux(settings,solar_rad,air_temp,rel_hum,water_temp,wind_speed,z,sed_type,bed_temp,depth_of_meas,shade,vts,cl,sol_refl,WP_m,width_m)
%   hflux_flux calculates the total heat entering and leaving a stream 
%   for a width of stream section over time in a way that can be called by 
%   hflux.m.  
%   The net heat flux is a function of shortwave radiation, 
%   longwave radiation, evaporation (latent heat), sensible heat, and 
%   bed conduction.
%   
% Usage:
%   [net shortwave longwave atm back land latent sensible bed] = 
%   hflux_flux(solar_rad,air_temp,rel_hum,water_temp,wind_speed,z,
%   sed_type,bed_temp,depth_of_meas,shade,vts,cl,sol_refl)
%
% Input:
%   Note: must all be for the same time period and distance (ie same size)
%   settings = an array of values specifying solution methods
%   solar_rad = an array of values for solar radiation (W/m^2) 
%   air_temp = an array of air temperature values (deg C) 
%   rel_hum = an array of relative humidity values (unitless)
%   water_temp = an array of values for stream temperature (deg C)
%   wind_speed = an array of wind speed values (m/s)
%   z = elevation of station where met data was obtained (m)
%   bed_temp=array of bed temperatures
%   depth_of_meas=distance between stream temp measurement and stream bed
%   temperature
%   shade = amount of shade (0-1)
%   cl = cloud cover (0-1)
%
% Output:
%  [net shortwave longwave atm back land latent sensible bed] = 
%  heat fluxes to be employed by hflux

% Example:
%  [net shortwave longwave atm back land latent sensible bed] = 
%  hflux_flux(settings,solar_rad,air_temp,rel_hum,water_temp,wind_speed,z,
%  sed_type,bed_temp,depth_of_meas,shade,vts,cl,sol_refl)

%   hflux_flux will call: 
%   1. hflux_shortwave: Calculate shortwave radiation
%   2. hflux_longwave: Calculate longwave radiation(atmospheric radiation,
%   back radiation, and land cover radiation) being recieved by water at 
%   each time step using the Stefan-Boltzman Law.
%       -calculate the atmospheric longwave radiation using
%   three inputs: air temperature, cloud cover, and relative humidity.  
%   Based on these two inputs, the emissivity of the atmosphere, actual 
%   vapor pressure, saturation vapor pressure will be determined.  
%       -calculate back radiation from the stream surface
%   using the water temperature and the Stefan-Boltzman constant. 
%       -calculate the landcover longwave radiation using air temperature, 
%   the Stefan-Boltzman constant, and shade
%   3. hflux_latent: Calculate latent heat flux using:
%       1. Penman combination method for open water
%       2. Mass-transfer method 
%   4. hflux_sensible: Calculate sensible heat flux, which is the heat exchange between
%   water and air and is driven by temperature differences, using:
%       1. Bowen ratio 
%       2. Direct method
%   5. hflux_bed: Calculate the bed conduction based on the change in temperature from
%   the stream to the bed and a thermal conductivity value assigned based
%   on the sediment type
%   
%   Variables:
%   INPUT
%   settings: values specifying solution options
%   solar_rad: solar radiation values (W/m^2)
%   sol_refl: reflected solar radiation (W/m^2)
%   air_temp: air temperature values in deg C
%   rel_hum: relative humidity values (unitless)
%   wind_speed: wind speed values (m/s)
%   water_temp: stream temperature (deg C)
%   sed_type: value or array of type 'cell' that described the sediment on
%   the stream bed
%   bed_temp: temperature of the stream bed (deg C)
%   depth_of_meas: depth at which bed_temp measurements are taken (m) 
%   shade: shade values along the stream ranging from 0 (total shade) to
%   1 (no shade)
%   vts: view to sky coefficient from (0 being no view to
%   sky, 1 being total view to sky)
%   cl: cloud cover ranging from 0 (clear) to 1 (overcast)
%   WP_m: wetted perimeter (m)
%   CALCULATED/CONSTANTS
%   e_s: saturation vapor pressure (kPa)
%   e_a: actual vapor pressure (kPa)
%   E_atm: emissivity of the atmosphere (unitless)
%   s_b: Stefan-Boltzman constant (W/m^2*C^4)
%   s: slope of the saturated vapor pressure (kPa/deg C)
%   r_a: aerodynamic resistance (s/m)
%   L_e: latent heat of vaporization (J/kg)
%   E: Penman open water evaporation (m/s)
%   rho_air: density of air (kg/m^3)
%   rho_water: density of water (kg/m^3)
%   c_air: specific heat capacity of air (J/kg*deg C)
%   c_water: specific heat capacity of water (J/kg*deg C)
%   psy: psychrometric constant (kPa/deg C)
%   ews: saturated vapor pressure (kPa)
%   ewa: actual vapor pressure (kPa)
%   z: elevation of site above sea level (m); adjust within code for
%   specific site
%   Pa: adiabatic atmospheric pressure (kPa)
%   fw: wind function (1/kPa)
%   b0: wind function constant (1/(m/s*kPa))
%   b1: wind function constant (%1/kPa)
%   Br: Bowen ratio (unitless)
%   K_sed: thermal conductivity of the stream bed sediment (W/m*C)
%   KH: coefficient for sensible heat transfer (J/C deg m3)
%   DH_DM: ratio of diffusivity of sensible heat to diffusivity
%   k: dimensionless constant
%   z_0: roughness heigh (m)
%   z_d: zero-plane displacement (m)
%   z_veg: height of vegetation (m)

% References: 
% Boyd, M., and Kasper, B., 2003, Analytical methods for dynamic open channel 
%   heat and mass Transfer: Methodology for heat source model Version 7.0:.
% Dingman, S.L., 1994, Physical Hydrology: Macmillan Publishing Company, New York.
% Holbert, K.E., 2007, Solar calculations: , no. 4, p. 1?7.
% Kustas, W., Rango, A., and Uijlenhoet, R., 1994, A simple energy budget 
%   algorithm for the snowmelt runoff model: Water Resources Research, v. 30, no. 5.
% Lapham, W., 1989, Use of Temperature Profiles Beneath Streams to Determine 
%   Rates of Vertical Ground-Water Flow and Vertical Hydraulic Conductivity:, 1?44 p.
% Loheide, S.P., and Gorelick, S.M., 2006, Quantifying stream-aquifer 
%   interactions through the analysis of remotely sensed thermographic profiles
%   and in situ temperature histories.: Environmental science & technology, 
%   v. 40, no. 10, p. 3336?41.
% Magnusson, J., Jonas, T., and Kirchner, J.W., 2012, Temperature dynamics 
%   of a proglacial stream: Identifying dominant energy balance components 
%   and inferring spatially integrated hydraulic geometry: Water Resources 
%   Research, v. 48, no. 6, p. 1?16, doi: 10.1029/2011WR011378.
% Maidment, D.R. (Ed.), 1993, Handbook of Hydrology: McGraw-Hill.
% Neumann, G., and Pierson, W., 1966, Principles of Physical Oceanography: 
%   Englewood Cliffs.
% Ouellet, V., Secretan, Y., Saint-Hilaire, A., and Morin, J., 2012, 
%   Water temperature modelling in a controlled environment: comparative 
%   study of heat budget equations: Hydrological Processes,, p. n/a?n/a, doi: 10.1002/hyp.9571.
% Recktenwald, G., 2011, Finite-difference approximations to the heat 
%   equation: Class Notes, v. 0, no. x.
% Webb, B., and Zhang, Y., 1997, Spatial and seasonal variability in 
%   the components of the river heat budget: Hydrological Processes, v. 11, p. 79?101.
% Westhoff, M.C., Savenije, H.H.G., Luxemburg, W.M.J.., Stelling, G.S., 
%   Van de Giesen, N.C., Selker, J.S., Pfister, L., and Uhlenbrook, S., 2007, 
%   A distributed stream temperature model using high resolution temperature 
%   observations: Hydrology and Earth System Sciences Discussions, v. 4, 
%   no. 1, p. 125?149, doi: 10.5194/hessd-4-125-2007.

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 1/19/2016

% Choose which equations to use:
eq1=settings(2,1); % Select shortwave radiation method
% 1 = Equation [3] in text, includes correction for reflection
% 2 = Equation [4] in text, includes albedo correction
eq2=settings(3,1); % Select latent heat flux equation
% 1 = Equation [14] in text, Penman equation
% 2 = Equation [17] in text, mass transfer method
eq3=settings(4,1); % Select sensible heat equation
% 1 = Equation [20] in text, Bowen ratio method
% 2 = Equation [24] in text, from Dingman, 1994
%%   SHORTWAVE RADIATION
[shortwave]=hflux_shortwave(solar_rad,shade,sol_refl,eq1);

%%   LONGWAVE RADIATION
% Calculate longwave radiation
% Westoff, et al. 2007/Boyd and Kasper, 2003
[longwave, atm_rad, back_rad, land_rad]=hflux_longwave(air_temp,water_temp,rel_hum,cl,vts);

%%   LATENT HEAT
% Calculate energy used for evaporation (latent heat) using the Penman
% equation for open water
[latent]=hflux_latent(water_temp,air_temp,rel_hum,wind_speed,shortwave,longwave,z,eq2);

%%   SENSIBLE HEAT
% Calculate the sensible heat flux, which is the heat exchange between
% the water and the air (driven by temperature differences)
[sensible]=hflux_sensible(water_temp,air_temp,rel_hum,wind_speed,z,latent,eq3);

%% STREAMBED CONDUCTION
%  Calculate the heat flux through the streambed
[bed]=hflux_bed(sed_type,water_temp,bed_temp,depth_of_meas,width_m,WP_m);

%% OUTPUT HEAT FLUXES
net=(shortwave+longwave+latent+sensible+bed)*60; %J/min*m^2 
shortwave; %W/m^2
longwave; %W/m^2
atm=atm_rad; %W/m^2
land=land_rad; %W/m^2
back=back_rad; %W/m^2
latent; %W/m^2
sensible; %W/m^2
bed; %W/m^2
end %function

% Copyright (c) 2013, AnneMarie Glose.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% (1) Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer.
% (2) Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.