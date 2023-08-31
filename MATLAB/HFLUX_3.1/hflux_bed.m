%% BED CONDUCTION CALCULATIONS
% Inputs:
% hflux_bed.m calculates the heat flux through the stream bed 
% Inputs:
% sed_type = a single value or array of type 'cell' that describe the
%            sediment type as clay, sand, gravel, or cobbles
% water_temp: water tempature
% bed_temp: stream bed temprature measurements
% depth_of_meas: depth below the stream bed that temperature measurements
% were collected
% width_m: width of the stream (meters)
% WP_m: wetted perimeter of the stream bed (meters)
% Output:
% bed: the heat flux through the stream bed

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014
%% Modified from Westhoff et al., 2007
function [bed]=hflux_bed(sed_type,water_temp,bed_temp,depth_of_meas, width_m, WP_m)
% Determine the sediment thermal conductivity (values from Lapham, 1989)
K_sed=zeros(length(sed_type),1);
for n=1:length(sed_type)
    if sed_type(n,1)==1
        K_sed(n,1)=0.84; %(W/m*C)
    elseif sed_type(n,1)==2
        K_sed(n,1)=1.2; %(W/m*C)
    elseif sed_type(n,1)==3
        K_sed(n,1)=1.4; %(W/m*C)
    elseif sed_type(n,1)==4
        K_sed(n,1)=2.5; %(W/m*C)
    end
end
% Calculate the heat flux through the stream bed
bed=(WP_m./width_m).*(-K_sed.*((water_temp-bed_temp)./depth_of_meas));
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