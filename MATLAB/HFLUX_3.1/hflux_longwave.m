%% LONGWAVE RADIATION CALCULATIONS

% Calculates the longwave radiation heat flux to be used by hflux_flux

% Input:
%   Note: must all be for the same time period and distance (ie same size)
%   air_temp = an array of air temperature values (deg C) 
%   rel_hum = an array of relative humidity values (unitless)
%   water_temp = an array of values for stream temperature (deg C)
%   vts = view to sky coefficient (0-1)
%   cl = cloud cover (0-1)
%
% Output:
%  [longwave atm_rad back_rad land_rad] = 
%  longwave radiation fluxes to be employed by hflux

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

%% Westoff, et al. 2007/ Boyd and Kasper, 2003/ Maidment, 2000/Kustas and Rango, 1994
function [longwave atm_rad back_rad land_rad]=hflux_longwave(air_temp,water_temp,rel_hum,cl,vts)
% Calculate saturation vapor pressure
e_s=0.611.*exp((17.27.*air_temp)./(237.2+air_temp)); 
% Calculate actual vapor pressure
e_a=(rel_hum./100).*e_s; 
% Calculates emissivity of atmosphere (Boyd and Kasper, 2003)
E_atm=1.72*((e_a./(air_temp+273.2)).^(1/7)).*(1+0.22*cl.^2); 
% Stefan-Boltzman constant (W m^-2 K^-4)
s_b=5.67E-8; 
% Calculate atmospheric radiation
atm_rad=0.96.*E_atm.*vts.*s_b.*((air_temp+273.2).^4);

% Calculate back radiation from the water column
back_rad=-0.96*s_b.*((water_temp+273.2).^4);

% Calculate land cover radiation
land_rad=0.96*(1-vts)*0.96*s_b.*((air_temp+273.2).^4);

% Calculate total longwave radiation
longwave=(atm_rad+back_rad+land_rad);
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