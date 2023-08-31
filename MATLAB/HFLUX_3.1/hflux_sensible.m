%% SENSIBLE HEAT CALCULATIONS

%   Calculates the sensible heat flux to be used by hflux_flux

% Input:
%   Note: must all be for the same time period and distance (ie same size)
%   air_temp = an array of air temperature values (deg C) 
%   rel_hum = an array of relative humidity values (unitless)
%   water_temp = an array of values for stream temperature (deg C)
%   wind_speed = an array of wind speed values (m/s)
%   z = elevation of station where met data was obtained (m)
%   latent = latent heat fulx (W/m^2)
%   eq3 = case1 calculates the sensible heat transfer based on the Bowen ratio heat flux
%         case2 calculates sensible heat transfer based on temperature differences,
%               Dingman, 1994
%               *This switch is set in hflux_flux.m

% Output:
%  [latent] = sensible heat flux (W/m^2)

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

function [sensible]=hflux_sensible(water_temp,air_temp,rel_hum,wind_speed,z,latent,eq3)
switch eq3
    case 1
        % Sensible heat transfer based on the Bowen ratio
        % Maidment, 1993, Magnusson et al., 2012, Westhoff et al., 2007, 
        % Boyd and Kasper, 2003
        % Calculate the saturated vapor pressure using the stream temperature
        ews=0.61275*exp((17.27*water_temp)./(237.3+water_temp)); %kPa
        % Calculate the actual vapor pressure using the stream temperature
        ewa=(rel_hum/100).*ews; %kPa
        % Calculate air pressure (kPa)
        Pa=101.3*(((293-(0.0065*z))/293)^5.256);
        % Calculate Bowen's ratio
        Br=0.00061*Pa.*((water_temp-air_temp)./(ews-ewa));
        % Calculate the sensible heat flux
        sensible=Br.*latent;
    case 2
        % Sensible heat transfer based on temperature differences,
        % Dingman, 1994 
        % Assign a value for the heat capacity of air
        c_air= 1004; %(J/kg deg C)
        % Assign a value for the density of air at 20 deg C
        rho_air=1.2041; %(kg/m^3)
        % Set the height of vegetation around stream
        z_veg=0.25; %m
        % Set the height at which the wind speed measurements were taken
        z_met=0.5; %m
        % Calculate height values from z_veg
        zd=0.7*z_veg; %zero-plane displacement, m
        z0=0.1*z_veg; %roughness height, m
        DH_DM=1; % ratio of diffusivity of sensible heat to diffusivity
        % of momentum, m2/s (1 under stable conditions)
        k=0.4; %dimensionless constant
        % Calculate KH
        KH=DH_DM*c_air*rho_air*(k^2/(log((z_met-zd)/z0))^2); %J/C deg m3
        sensible=-KH*wind_speed.*(water_temp-air_temp); %W/m2
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