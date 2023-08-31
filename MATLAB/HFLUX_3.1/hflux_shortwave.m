%% SHORTWAVE RADIATION CALCULATIONS

% Calculates the shortwave radiation flux to be used by hflux_flux

% Inputs:
%   solar_rad_in = total incoming solar radiation data at each time_met
%       (program interpolates in time)
%   shade = values for shading (0 to 1, with 0 being min shading and 1 being max shading)
%   sol_refl = portion of solar radiation that is reflected off the surface
%   of the stream
%   eq1 = case1 uses the Ouellet, et al. 2012 and Boyd and Kasper,
%         2003 methods
%         case2 uses the Magnusson, et al. 2012 nethod
%               *This switch was set in hflux_flux.m

% Outputs:
% [shortwave] = shortwave radiation (W/m^2)

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

function [shortwave]=hflux_shortwave(solar_rad,shade,sol_refl,eq1)
switch eq1
    case 1
        % Ouellet, et al. 2012 and Boyd and Kasper, 2003
        % Correct the incoming solar radiation measured at the site for shading
        sol_in=(1-shade).*solar_rad;
        
        % Calculate total shortwave radiation
        shortwave=sol_in-(sol_refl*sol_in);
    case 2
        % Magnusson, et al. 2012
        % Calculate shortwave radiation by correcting the measured values for
        % albedo
        albedo=0.05;
        shortwave=(1-albedo).*((1-shade).*solar_rad);
end
end
