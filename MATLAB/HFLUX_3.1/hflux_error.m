function [rel_err, me, mae, mse, rmse, nrmse]=hflux_error(time_mod,dist_mod,temp_mod,dist_temp,time_temp,temp,unattend)
% HFLUX ERROR ANALYSIS
% The HFLUX Error Analysis is an add-on to the HFLUX Stream
% Temperature Solver.
%
% Description:
% hflux_error calculates a suite of error metrics using the modeled and
% measured stream temperature data.  The program plots the mean modeled
% and measured stream temperature with respect to space and time.
%
% USAGE:
% [rel_err me mae mse rmse nrmse]=hflux_error(time_mod,dist_mod,temp_mod,dist_temp,time_temp,temp)
% with optional switch for "unattended mode":
% [rel_err me mae mse rmse nrmse]=hflux_error(time_mod,dist_mod,temp_mod,dist_temp,time_temp,temp,'unattended')

% Inputs:
% time_m    = temporal resolution of modeled data
% dist_m    = location of modeled data points
% temp_mod  = modeled temperatures
% dist_temp = location of measured data points
% time_temp = time scale of measured data
% temp      = measured temperature data
% unattend  = an optional switch to control user interaction with hflux.
%            Enter the text string 'unattended' for this input variable to
%            supress all pauses and user-input questions during normal
%            operation.
% Outputs:
% rel_err = relative error
% me = mean error
% mae = mean absolute error
% mse = mean square error
% rmse = root mean square error
% nrmse = normalized root mean square error

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

% set up switch to turn off displays and user inputs
if nargin==6, unattend='no'; end %create 'unattend' input argument if not entered in function call
unattend=strcmpi(unattend,'unattended'); %set unattend to logical 1 if input as 'unattended', logical 0 otherwise

if ~unattend
    disp('Checking input arguments...')
end

% Check input arguments
if nargin<6 %if the number of input arguments is incorrect
    error('Wrong number of input arguments.')
elseif nargin>7 %if the number of input arguments is incorrect
    error('Wrong number of input arguments.')
elseif isvector(time_mod)+isvector(dist_mod)+isvector(dist_temp)+isvector(time_temp)~=4 %if time_mod, dist_mod, time_temp, and dist_temp are not column vectors
    error('time_mod, dist_mod, dist_temp, and time_temp must all be column vectors.')
elseif ismatrix(temp_mod)+ismatrix(temp)~=2 %if temp_mod and temp are not matrices
    error('temp_mod and temp must be matrices.')
end %if

if ~unattend
    disp('...Done!')
    disp('    ')
    disp('Resampling data to original spatial and temporal resolution...')
end

% Resample the modeled data to the same temporal and spatial scales as the
% measured temperature data
for n=1:(length(time_mod))
    temp_dx(:,n)=interp1(dist_mod(:,1),temp_mod(:,n),dist_temp(:,1));
end
for n=1:length(dist_temp)
    temp_dt(n,:)=interp1(time_mod(:,1),temp_dx(n,:),time_temp(:,1));
end

if ~unattend
    disp('...Done!')
    disp('    ')
    disp('Calculating error metrics...')
end

% Calculate the percent relative error
rel_err=((temp-temp_dt)./temp)*100;
% Calculate the mean residual error
me=sum(temp(:)-temp_dt(:))/numel(temp);
% Calculate the mean absolute residual error
mae=sum(abs(temp(:)-temp_dt(:)))/numel(temp);
% Calculate the mean square error
mse=sum((temp(:)-temp_dt(:)).^2)/numel(temp);
% Calculate the root mean square error
rmse=sqrt(sum((temp(:)-temp_dt(:)).^2)/numel(temp));
% Calculate the normalized root mean square error
nrmse=(rmse/(max(temp(:))-min(temp(:))))*100;
% Calculate the Nash-Sutcliffe Efficiency
% nse=1-(sum((temp(:)-temp_dt(:)).^2)/sum((temp(:)-mean(temp_dt(:))).^2));

if ~unattend
    disp('...Done!')
    disp('     ')
end

if ~unattend
    
    % 2D plot of temperature residuals
    f0 = figure('Visible','on');
    residuals=temp-temp_dt;
    imagesc(time_temp(:,1),dist_temp(:,1),residuals);
    colormap(jet(256));
    t=colorbar;% Inserts a colorbar. a handle is created
    set(get(t,'ylabel'),'string',['Model Residuals (' char(176) 'C)'],...
        'FontSize',12,'FontWeight','Bold')
    axis square;
    plot_title='Modeled Temperature Residuals';
    font_type='Arial';
    xlab='Time (min)';
    ylab='Distance (m)';
    set(gca,'FontName','Arial');
    title(plot_title,...
        'FontName' , font_type, ...
        'FontSize'   , 12 , 'FontWeight','Bold');
    xlabel(xlab,...
        'FontName' , font_type, ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    ylabel(ylab,...
        'FontName' , font_type, ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    
    f1 = figure('Visible','on');
    subplot(2,1,1)
    plot(dist_temp(:,1),mean(temp,2),'--ko','LineWidth',1.5,...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','b',...
        'MarkerSize',8)
    hold on
    plot(dist_mod,mean(temp_mod,2),'r','LineWidth',1.5)
    set(gca,'FontName','Arial')
        set(gca,'XLim',[min(dist_temp),...
        max(dist_temp)])
    a=min(mean(temp,2));
    b=min(mean(temp_mod,2));
    c=max(mean(temp,2));
    d=max(mean(temp_mod,2));
    set(gca,'YLim',[min([a b]) max([c d])])
    title('Stream Temperature Along the Reach',...
        'FontName' , 'Arial', ...
        'FontSize'   , 14,'FontWeight','Bold'  );
    xlabel('Distance Downstream (m)',...
        'FontName' , 'Arial', ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    ylabel(['Temperature (' char(176) 'C)'],...
        'FontName' , 'Arial', ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    legend('Measured','Modeled','Location','Best');
    
    
    subplot(2,1,2)
    plot(time_temp(:,1),mean(temp),'b','LineWidth',1.5)
    hold on
    plot(time_mod,mean(temp_mod),'r','LineWidth',1.5)
    set(gca,'XLim',[min(time_temp),...
        max(time_temp)]);
    a=min(mean(temp,1));
    b=min(mean(temp_mod,1));
    c=max(mean(temp,1));
    d=max(mean(temp_mod,1));
    set(gca,'YLim',[min([a b])-1, max([c d])+1]);
    set(gca,'FontName','Arial');
    title('Stream Temperature Over Time',...
        'FontName' , 'Arial', ...
        'FontSize'   , 14 ,'FontWeight','Bold' );
    xlabel('Time (min)',...
        'FontName' , 'Arial', ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    ylabel(['Temperature (' char(176) 'C)'],...
        'FontName' , 'Arial', ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    legend('Measured','Modeled','Location','Best');
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