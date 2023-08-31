function [output]=hflux_bed_sed(sed_type,dist_bed,dist_mod)

% bed_sed.m interpolates the bed sediment input used by hflux_flux so
% there is a description of the type of sediment at each node
% Inputs:
% sed_type = a single value or array of type 'cell' that describe the
%            sediment type as clay, sand, gravel, or cobbles
% dist_bed = distances in meters where the sediment type was observed
% dist_mod = interpolated distances in meters used in the model
% Output:
% An array of numbers that represents the type of sediment of a size that
% corresponds with the model nodes that is used by hflux_flux.m

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

sed_type_num=zeros(length(sed_type),1);

if ischar(sed_type)==1
    if strcmp(sed_type,'clay')
        sed_type_num=1;
    elseif strcmp(sed_type,'sand')
        sed_type_num=2;
    elseif strcmp(sed_type,'gravel')
        sed_type_num=3;
    elseif strcmp(sed_type,'cobbles')
        sed_type_num=4;
    end
    output=sed_type_num;
elseif iscell(sed_type)==1
    for n=1:length(sed_type)
        if strcmp(sed_type(n,1),'clay')
            sed_type_num(n,1)=1;
        elseif strcmp(sed_type(n,:),'sand')
            sed_type_num(n,1)=2;
        elseif strcmp(sed_type(n,1),'gravel')
            sed_type_num(n,1)=3;
        elseif strcmp(sed_type(n,1),'cobbles')
            sed_type_num(n,1)=4;
        end
    end
    sed_type_int=interp1(dist_bed,sed_type_num,dist_mod,'nearest');
    output=sed_type_int;
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