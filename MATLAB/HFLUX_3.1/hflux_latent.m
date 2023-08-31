%% LATENT HEAT CALCULATIONS

% Calculates the latent heat flux to be used by hflux_flux

% Input:
%   Note: must all be for the same time period and distance (ie same size)
%   shortwave = an array of values for solar radiation (W/m^2) 
%   longwave = an array of values for longwave radiation (W/m^2) 
%   rel_hum = an array of relative humidity values (unitless)
%   water_temp = an array of values for stream temperature (deg C)
%   wind_speed = an array of wind speed values (m/s)
%   z = elevation of station where met data was obtained (m)
%   eq2 = case1 uses the Penman Method to calculate latent heat flux
%         case2 uses the Mass Transfer Method to calculate latent heat flux
%               *This switch is set in hflux_flux.m

% Output:
%  [latent] = latent heat flux 

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

function [latent]=hflux_latent(water_temp,air_temp,rel_hum,wind_speed,shortwave,longwave,z,eq2)
switch eq2
    case 1
        % PENMAN METHOD: Westhoff et al., 2007
        % Calculate saturation vapor pressure
        e_s=0.611.*exp((17.27.*air_temp)./(237.2+air_temp));
        % Calculate actual vapor pressure
        e_a=(rel_hum./100).*e_s;
        % Calculate the latent heat of vaporization
        L_e=1000000*(2.501-(0.002361.*water_temp));%J/kg
        % Calculate the slope of the saturated vapor pressure curve at a given air
        % temperature
        s=(4100.*e_s)./((237+air_temp).^2); %kPa/deg C
        % Calculate the aerodynamic resistance
        r_a=245./((0.54.*wind_speed)+0.5); %[s/m]
        % Assign a value for the heat capacity of air
        c_air= 1004; %(J/kg deg C)
        % Assign a value for the density of water
        rho_water=1000; %(kg/m^3)
        % Assign a value for the density of air at 20 deg C
        rho_air=1.2041; %(kg/m^3)
        % Calculate air pressure
        Pa=101.3-(0.0105*z);
        % Calculate psychrometric constant (kPa/deg C) (based on air
        % pressure (Pa), (value should be adjusted for different site elevations), ratio
        % of water to dry air=.622, and the latent heat of water vaporization=2.45E6
        % [J/kg deg C])
        psy= (c_air*Pa)/(.622*2450000);
        % Calculate the Penman open water evaporation
        E=(((s.*(shortwave+longwave))./(rho_water.*L_e.*(s+psy))))+((c_air.*rho_air.*psy.*(e_s-e_a))./(rho_water.*L_e.*r_a.*(s+psy)));
        % Calculate the latent heat flux
        latent=(-rho_water).*(L_e).*E;
    case 2
        % MASS TRANSFER METHOD: Boyd and Kasper, 2003
        % Calculate saturation vapor pressure of the air
        e_s=0.611.*exp((17.27.*air_temp)./(237.2+air_temp)); %kPa
        % Calculate actual vapor pressure of the air
        e_a=(rel_hum./100).*e_s; %kPa
        % Calculate the saturated vapor pressure at the stream surface
        ews=0.611*exp((17.27*water_temp)./(237.3+water_temp)); %kPa
        % Calculate the latent heat of vaporization
        L_e=1000000*(2.501-(0.002361.*water_temp));%J/kg
        % Assign a value for the density of water
        rho_water=1000; %(kg/m^3)
        % Set coefficients for wind function
        b0=1.505*10^-8; %1/(m/s*kPa)
        b1=1.6*10^-8; %1/kPa
        % Calculate the wind function
        fw=b0+(b1*wind_speed); %(1/kPa)
        % Calculate the evaporation rate
        E=fw.*(ews-e_a); %m/s
        % Calculate the latent heat flux
        latent=-E.*L_e*rho_water; %W/m2
end
end

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