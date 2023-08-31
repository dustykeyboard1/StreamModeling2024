function [input_data]=hflux_format(input_file_name)
% hflux_format reads in input data for use in hflux.m from an Excel sheet.
% hflux_format checks the input data to ensure it is in the correct format
% for use by hflux.m The function outputs a structure called input_data
% that is the input for the hflux function.
%
% USAGE: [input_data]=hflux_format(input_file_name)
%
% INPUT:
% input_file_name: Microsoft Excel sheet with necessary input data
%
% OUTPUT:
% input_data: structure containing the input data to be used by hflux.m

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 1/19/2016
%% Read in input data from Excel spreadsheet
hfluxversion='HFLUX Beta';
disp(' ')
disp(hfluxversion)
disp(' ')
disp('Reading in data...')
settings=xlsread(input_file_name,'settings'); 
%settings=readtable(input_file_name,'Sheet','settings');% was xlsread, also added keyword 'Sheet'
time_mod=xlsread(input_file_name,'time_mod');
dist_mod=xlsread(input_file_name,'dist_mod');
temp_x0_data=xlsread(input_file_name,'temp_x0_data');
temp_t0_data=xlsread(input_file_name,'temp_t0_data');
dim_data=xlsread(input_file_name,'dim_data');
dis_data=xlsread(input_file_name,'dis_data');  
time_dis=xlsread(input_file_name,'time_dis'); %new
met_data=xlsread(input_file_name,'met_data');
bed_data1=xlsread(input_file_name,'bed_data1');
bed_data2=xlsread(input_file_name,'bed_data2');
[~, ~, sed_type]=xlsread(input_file_name,'sed_type');
T_L_data=xlsread(input_file_name,'T_L_data');
shade_data=xlsread(input_file_name,'shade_data');
cloud_data=xlsread(input_file_name,'cloud_data');
site_info=xlsread(input_file_name,'site_info');
disp('Done!')

%% Check input arguments
% Build a vector of column sizes for the matrix inputs in order to check
% their sizes against what is required by hflux.m

disp('Checking input arguments...')
[~,col]=size(temp_x0_data);
ncol_input(1,1)=col;
[~,col]=size(temp_t0_data);
ncol_input(1,2)=col;
[~,col]=size(dim_data);
ncol_input(1,3)=col;
[~,col]=size(T_L_data);
ncol_input(1,5)=col;
[~,col]=size(met_data);
ncol_input(1,6)=col;
[~,col]=size(bed_data1);
ncol_input(1,7)=col;
[~,col]=size(shade_data);
ncol_input(1,8)=col;
[~,col]=size(cloud_data);
ncol_input(1,9)=col;
[~,col]=size(site_info);
ncol_input(1,10)=col;

if nargin<1 %if the number of input arguments is too small
    error('Too few input arguments.')
elseif nargin>2 %if the number of input arguments is too large
    error('Too many input arguments.')
elseif ncol_input(1,1)~=2 % if the matrix inputs have the wrong number of columns
    error('temp_x0_data must contain 2 columns of data')
elseif ncol_input(1,2)~=2 % if the matrix inputs have the wrong number of columns
    error('temp_t0_data must contain 2 columns of data')
elseif ncol_input(1,3)~=5 % changed to 5 from 4
    error('dim_data must contain 5 columns of data')
elseif ncol_input(1,5)~=2
    error('T_L_data must contain 2 columns of data')
elseif ncol_input(1,6)~=10
    error('met_data must contain 10 columns of data')
elseif ncol_input(1,7)~=2
    error('bed_data1 must contain 2 columns of data')
elseif ncol_input(1,8)~=3
    error('shade_data must contain 3 columns of data')
elseif ncol_input(1,9)~=2
    error('cloud_data must contain 2 columns of data')
elseif ncol_input(1,10 )~=1
    error('site_info must contain 1 column of data')
elseif isvector(time_mod)+isvector(dist_mod)~=2 %if time_mod and dist_mod are not column vectors
    error('time_mod and dist_mod must be column vectors.')
elseif iscell(sed_type)==0 %if sed_type is not a cell array
    error('sed_type must be a cell array')
end %if
disp('Done!')

%% Output structure (for use by hflux.m)
input_data.settings=settings;
input_data.time_mod=time_mod;
input_data.dist_mod=dist_mod;
input_data.temp_t0_data=temp_t0_data;
input_data.temp_x0_data=temp_x0_data;
input_data.dim_data=dim_data;
input_data.dis_data=dis_data;
input_data.time_dis=time_dis; 
input_data.met_data=met_data;
input_data.bed_data1=bed_data1;
input_data.bed_data2=bed_data2;
input_data.sed_type=sed_type;
input_data.T_L_data=T_L_data;
input_data.shade_data=shade_data;
input_data.cloud_data=cloud_data;
input_data.site_info=site_info;

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