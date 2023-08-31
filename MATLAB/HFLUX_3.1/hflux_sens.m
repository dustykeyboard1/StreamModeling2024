function [sens]=hflux_sens(input_data,dis_high_low,T_L_high_low,vts_high_low,shade_high_low)
% HFLUX SENSITIVITY ANALYSIS
% The HFLUX Sensitivity Analysis is an add-on to the HFLUX Stream
% Temperature Solver.
%
% DESCRIPTION:
%   Does a basic sensitivity analysis on HFLUX inputs that are uniform through
%   time and plots results.  Each input parameter is input as it is in HFLUX,
%   and then the program calculates
%   the range of flux using the range of each individual parameter (low and
%   high), while holding all others at their base values.
%
% USAGE:
%   sens = hflux_sens(time_mod,dist_mod,temp_x0_data,temp_t0_data,dim_data,
%   dis_data,dis_high_low,T_L_data,T_L_high_low,met_data,z,bed_data,
%   vts_high_low,sed_type,shade_data,shade_high_low)
%
% INPUT:
%   input_data: see "help hflux" for details
%   dis_high_low: a pair of values that determine the increase and decrease
%   in the discharge value desired by the user.  Enter values in square
%   brackets with the low value first, ie [-0.5 0.5]
%   T_L_high_low: a pair of values that determine the increase and decrease
%   in the temperature of lateral inflows desired by the user.  Enter values
%   in square brackets with the low value first, ie [-2 2]
%   vts_high_low: a pair of values that determine the increase and decrease
%   in the vts coefficient desired by the user.  Enter values
%   in square brackets with the low value first, ie [-0.2 0.2]
%   shade_high_low: a pair of values that determine the increase and decrease
%   in the shading desired by the user.  Enter values
%   in square brackets with the low value first, ie [-0.2 0.2]
%
% EXAMPLE:
%   [sens]=hflux_sens(time_mod,dist_mod,temp_x0_data,temp_t0_data,dim_data,
%   dis_data,[-.01 .01],T_L_data,[-2 2],met_data,[-2 2],
%   shade_data,[-.2 .2])

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

disp('Calculating high and low values...')

% Set program option to suppress graphical output
input_data.settings(5)=1;

% Create low and high values for each parameter
dis_low=input_data.dis_data(:,2:end)+dis_high_low(1); %changed to work with more columns
dis_high=input_data.dis_data(:,2:end)+dis_high_low(2); %changed to work with more columns
T_L_low=input_data.T_L_data(:,2)+T_L_high_low(1);
T_L_high=input_data.T_L_data(:,2)+T_L_high_low(2);
vts_low=input_data.shade_data(:,3)+vts_high_low(1);
vts_high=input_data.shade_data(:,3)+vts_high_low(2);
shade_low=input_data.shade_data(:,2)+shade_high_low(1);
shade_high=input_data.shade_data(:,2)+shade_high_low(2);

% Create hflux-ready arrays from the low and high values
dis_data_low = horzcat(input_data.dis_data(:,1),dis_low); %allows for multiple columns of discharge data
dis_data_high = horzcat(input_data.dis_data(:,1),dis_high);
% dis_data_low(:,1)=input_data.dis_data(:,1); %previous code for no changing discharge
% dis_data_low(:,2)=dis_low;
% dis_data_high(:,1)=input_data.dis_data(:,1);
% dis_data_high(:,2)=dis_high;

T_L_data_low(:,1)=input_data.T_L_data(:,1);
T_L_data_low(:,2)=T_L_low;
T_L_data_high(:,1)=input_data.T_L_data(:,1);
T_L_data_high(:,2)=T_L_high;

vts_data_low(:,1)=input_data.shade_data(:,1);
vts_data_low(:,2)=input_data.shade_data(:,2);
vts_data_low(:,3)=vts_low;
vts_data_high(:,1)=input_data.shade_data(:,1);
vts_data_high(:,2)=input_data.shade_data(:,2);
vts_data_high(:,3)=vts_high;

shade_data_low(:,1)=input_data.shade_data(:,1);
shade_data_low(:,2)=shade_low;
shade_data_low(:,3)=input_data.shade_data(:,3);
shade_data_high(:,1)=input_data.shade_data(:,1);
shade_data_high(:,2)=shade_high;
shade_data_high(:,3)=input_data.shade_data(:,3);

% Overwrite values in input data for each case
input_data_base=input_data;
input_data_lowdis=input_data;
input_data_lowdis.dis_data=dis_data_low;
input_data_highdis=input_data;
input_data_highdis.dis_data=dis_data_high;
input_data_lowT_L=input_data;
input_data_lowT_L.T_L_data=T_L_data_low;
input_data_highT_L=input_data;
input_data_highT_L.T_L_data=T_L_data_high;
input_data_lowvts=input_data;
input_data_lowvts.shade_data=vts_data_low;
input_data_highvts=input_data;
input_data_highvts.shade_data=vts_data_high;
input_data_lowshade=input_data;
input_data_lowshade.shade_data=shade_data_low;
input_data_highshade=input_data;
input_data_highshade.shade_data=shade_data_high;

disp('...Done!')
disp('    ')
disp('Running HLUX for the base, high, and low cases...')

% Run hlux.m for middle (base) values, then for high and low values of
% each parameter with other parameters kept at base values
[temp_mod_base, ~, ~, ~]=hflux(input_data_base);
[temp_mod_lowdis, ~, ~, ~]=hflux(input_data_lowdis);
[temp_mod_highdis, ~, ~, ~]=hflux(input_data_highdis);
[temp_mod_lowT_L, ~, ~, ~]=hflux(input_data_lowT_L);
[temp_mod_highT_L, ~, ~, ~]=hflux(input_data_highT_L);
[temp_mod_lowvts, ~, ~, ~]=hflux(input_data_lowvts);
[temp_mod_highvts, ~, ~, ~]=hflux(input_data_highvts);
[temp_mod_lowshade, ~, ~, ~]=hflux(input_data_lowshade);
[temp_mod_highshade, ~, ~, ~]=hflux(input_data_highshade);

disp('...Done!')
disp('    ')
disp('Writing output data...')
% Write outputs from hflux to output structures
base.temp=temp_mod_base;
base.mean=mean(temp_mod_base,2);

lowdis.temp=temp_mod_lowdis;
lowdis.mean=mean(temp_mod_lowdis,2);

highdis.temp=temp_mod_highdis;
highdis.mean=mean(temp_mod_highdis,2);

lowT_L.temp=temp_mod_lowT_L;
lowT_L.mean=mean(temp_mod_lowT_L,2);

highT_L.temp=temp_mod_highT_L;
highT_L.mean=mean(temp_mod_highT_L,2);

lowvts.temp=temp_mod_lowvts;
lowvts.mean=mean(temp_mod_lowvts,2);

highvts.temp=temp_mod_highvts;
highvts.mean=mean(temp_mod_highvts,2);

lowshade.temp=temp_mod_lowshade;
lowshade.mean=mean(temp_mod_lowshade,2);

highshade.temp=temp_mod_highshade;
highshade.mean=mean(temp_mod_highshade,2);

% Write structures to output
sens.dis_l=dis_data_low;
sens.dis_h=dis_data_high;
sens.TL_l=T_L_data_low;
sens.TL_h=T_L_data_high;
sens.vts_l=vts_low;
sens.vts_h=vts_high;
sens.sh_l=shade_data_low;
sens.sh_h=shade_data_high;
sens.base=base;
sens.lowdis=lowdis;
sens.highdis=highdis;
sens.lowT_L=lowT_L;
sens.highT_L=highT_L;
sens.lowvts=lowvts;
sens.highvts=highvts;
sens.lowshade=lowshade;
sens.highshade=highshade;

disp('...Done!')
disp('    ')

% Make sensitivity plots
figure;
subplot(2,2,1)
hold on
plot(input_data.dist_mod,sens.lowdis.mean,'--b','LineWidth',2)
plot(input_data.dist_mod,sens.base.mean,'k','LineWidth',2)
plot(input_data.dist_mod,sens.highdis.mean,'--r','LineWidth',2)
hold off
set(gca,'XLim',[min(input_data.temp_t0_data(:,1)),...
    max(input_data.temp_t0_data(:,1))]) %set x-axis limits to dist range
set(gca,'FontName','Arial')
title('Discharge','FontName', 'Arial','FontSize',14,'FontWeight','Bold'), ...
    xlabel('Distance Downstream (m)','FontName', 'Arial','FontSize',...
    12,'FontWeight','Bold'),...
    ylabel(['Temperature (' setstr(176) 'C)'],'FontName', 'Arial',...
    'FontSize',12,'FontWeight','Bold')
legend('Low','Base','High'); %legend of values (base,high,low)

subplot(2,2,2)
hold on
plot(input_data.dist_mod,sens.lowT_L.mean,'--b','LineWidth',2)
plot(input_data.dist_mod,sens.base.mean,'k','LineWidth',2)
plot(input_data.dist_mod,sens.highT_L.mean,'--r','LineWidth',2)
hold off
set(gca,'XLim',[min(input_data.temp_t0_data(:,1)),...
    max(input_data.temp_t0_data(:,1))]) %set x-axis limits to dist range
set(gca,'FontName','Arial')
title('Groundwater Temperature','FontName', 'Arial','FontSize',14,'FontWeight','Bold'), ...
    xlabel('Distance Downstream (m)','FontName', 'Arial','FontSize',...
    12,'FontWeight','Bold'),...
    ylabel(['Temperature (' setstr(176) 'C)'],'FontName', 'Arial',...
    'FontSize',12,'FontWeight','Bold')
legend('Low','Base','High'); %legend of values (base,high,low)

subplot(2,2,3)
hold on
plot(input_data.dist_mod,sens.lowvts.mean,'--b','LineWidth',2)
plot(input_data.dist_mod,sens.base.mean,'k','LineWidth',2)
plot(input_data.dist_mod,sens.highvts.mean,'--r','LineWidth',2)
hold off
set(gca,'XLim',[min(input_data.temp_t0_data(:,1)),...
    max(input_data.temp_t0_data(:,1))]) %set x-axis limits to dist range
set(gca,'FontName','Arial')
title('View to Sky Coefficient','FontName', 'Arial','FontSize',14,'FontWeight','Bold'), ...
    xlabel('Distance Downstream (m)','FontName', 'Arial','FontSize',...
    12,'FontWeight','Bold'),...
    ylabel(['Temperature (' setstr(176) 'C)'],'FontName', 'Arial',...
    'FontSize',12,'FontWeight','Bold')
legend('Low','Base','High'); %legend of values (base,high,low)

subplot(2,2,4)
hold on
plot(input_data.dist_mod,sens.lowshade.mean,'--b','LineWidth',2)
plot(input_data.dist_mod,sens.base.mean,'k','LineWidth',2)
plot(input_data.dist_mod,sens.highshade.mean,'--r','LineWidth',2)
hold off
set(gca,'XLim',[min(input_data.temp_t0_data(:,1)),...
    max(input_data.temp_t0_data(:,1))]) %set x-axis limits to dist range
set(gca,'FontName','Arial')
title('Shade','FontName', 'Arial','FontSize',14,'FontWeight','Bold'), ...
    xlabel('Distance Downstream (m)','FontName', 'Arial','FontSize',...
    12,'FontWeight','Bold'),...
    ylabel(['Temperature (' setstr(176) 'C)'],'FontName', 'Arial',...
    'FontSize',12,'FontWeight','Bold')
legend('Low','Base','High'); %legend of values (base,high,low)

% Calculate total percent change in stream temperature
change=[mean(mean((sens.lowdis.temp))-mean(mean(sens.base.temp))),...
    (mean(mean(sens.highdis.temp))-mean(mean(sens.base.temp)));...
    (mean(mean(sens.lowT_L.temp))-mean(mean(sens.base.temp))),...
    (mean(mean(sens.highT_L.temp))-mean(mean(sens.base.temp)));...
    (mean(mean(sens.lowvts.temp))-mean(mean(sens.base.temp))),...
    (mean(mean(sens.highvts.temp))-mean(mean(sens.base.temp)));...
    (mean(mean(sens.lowshade.temp))-mean(mean(sens.base.temp))),...
    (mean(mean(sens.highshade.temp))-mean(mean(sens.base.temp)))];
figure;
colormap(hot);
bar(change,'grouped')
labels={'Discharge';'GW Temp';'VTS';'Shade'};
title('Change in Average Stream Temperature With Changing Variables',...
    'FontName','Arial','FontSize',14,'FontWeight','Bold');
ylabel(['Change (' char(176) 'C)'],'FontName','Arial','FontSize',12,'FontWeight','Bold');
xlabel('Adjusted Variable','FontName','Arial','FontSize',12,'FontWeight','Bold');
set(gca,'xticklabel',labels,'FontName','Arial','FontSize',12);
legend('Decrease Value','Increase Value','Location','Best');


sens.change=change;
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