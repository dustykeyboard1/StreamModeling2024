function [temp_mod, matrix_data, node_data, flux_data]=hflux(input_data)
        
%% HFLUX STREAM TEMPERATURE SOLVER
% HFLUX is a program that uses a deterministic modeling approach to calculate stream
% temperature with respect to space and time.  The program uses
% hydrological information, meteorological data, and site information as
% input data and solves for stream temperature using a Crank-Nicholson
% finite difference scheme or the second order Runge Kutta method.  See HFLUX 
% documentation for more information regarding methodology and running the program.
%
% USAGE:
%   [temp_mod matrix_data node_data flux_data] = hflux(input_data)
%
% UNITS OF INPUT FILES:
% meters, minutes, C, m3/s (program converts to m3/min), m2
% %
% INPUTS:   
% A structure created by hflux_format (input_data) containing the following information
% is the input to HFLUX.
% settings = values to set finite difference solution scheme, and select
%   equations for solving for shortwave radiation, longwave heat flux, and
%   shortwave heat flux. User also selects whether to display output
% time_mod = model times at which temperatures will be computed for heat
%   budget, in minutes
% dist_mod = model distances along the reach where temperature will be
%   calculated, in m
% TEMP_X0_DATA (array with columns arranged 1 to 2, left to right)
%   1. time_temp = times at which temperature measurements are known, in minutes
%   2. temp_x0 = upstream boundary condition at each time_temp, in C (program
%       will interpolate in time)
% TEMP_T0_DATA (array with columns arranged 1 to 2, left to right)
%   1. dist = distances along the reach where linitial temperature data is given,
%       in meters
%   2. temp_t0 = observed temperatures at each dist at time=0, in C (initial
%       condition; program interpolates in space)
% DIM_DATA (array with columns arranged 1 to 4, left to right)
%   1. dist_stdim = distances along the reach where stream dimensions were
%       measured, in meters
%   2. area = known area of the stream channel, in m2, along the reach at each
%       dist (program interpolates in space)
%   3. width = stream width at each dist in meters (program interpolates in
%       space)
%   4. depth = stream depth at each dist in meters (program interpolates in
%       space)
%   5. discharge_stdim = discharge rate at which stream dimension data was
%       observed
% DIS_DATA (array with columns arranged left to right)
%   1. dist_dis = distances along the reach where discharge measurements were
%       taken, in meters (column 1)
%   2. discharge = known discharge rates, in m3/s, along the reach at each
%       dist; each column represents the discharge at a different time
%       (columns 2 through the number of times discharge was measured)
% time_dis = vector containing the times at which the discharge was measured in minutes                 
%       (program interpolates in space and converts to m3/min)
% T_L_DATA
%   1. dist_T_L = distances at which groundwater temperature is known (m)
%   2. T_L = temperature of lateral groundwater discharge at each dist, in C
%      (program interpolates in space)
% MET_DATA (array with columns arranged 1 to 10, left to right)
%   1. year = year during which measurements were taken (ie. 2012)
%   2. month = month during which measurements were taken (ie. 6)
%   3. day = day during which measurements were taken (ie. 18)
%   4. hour = hour during which measurements were taken (ie. 22) (military
%      time)
%   5. min = minute during which measurements were taken (ie. 5)
%   6. time_met = times at which meteorological data are known, in minutes (met
%       data includes air temp, rel humidity, wind speed, and solar rad)
%   7. solar_rad_in = total incoming solar radiation data at each time_met
%       (program interpolates in time)
%   8. air_temp_in = air temperature data at each time_met
%       (program interpolates in time)
%   9. rel_hum_in = relative humidity data at each time_met
%       (program interpolates in time)
%   10. wind_speed_in = wind speed data at each time_met
%       (program interpolates in time)
% BED_DATA1 (array with columns arranged 1 to 2, left to right)
%   1. dist_bed = distances (m) at which bed temperatures were taken
%   2. depth_of_meas = depth, in m, at which bed temperature measurements
%       were made
% BED_DATA2 (array with rows arranged 1 to # of dist_bed, top to bottom)
%   1. time_bed = times (min) at which bed temperatures were taken
%   2. bed_temp = the temperature in the streambed, in C, at each time (along rows)
%   and space (along columns) steps at the depth given in depth_of_measurement
%   (program interpolates in space and time)
% sed_type = general sediment classification ('clay', 'silt', 'sand', orF
%   'gravel')   *must be same spatial scale as BED_DATA
% SHADE_DATA (array with columns arranged 1 to 3, left to right)
%   1. dist_shade = distances (m) at which shade values were observed
%   2. shade = values for shading (0 to 1, with 0 being min shading and 1 being max shading)
%   3. vts = values for the view to sky coefficient (0 being no view to
%   sky, 1 being total view to sky)
% CLOUD_DATA (array with columns arranged 1 to 2, left to right)
%   1. time_cloud = times (min) at which cloud values were observed
%   2. c_in = values for cloud cover (0 to 1, with 0 being clear and 1 being overcast)
% SITE_INFO (column with three rows)
%   1. lat = latitude (positive for northern hemisphere, negative for southern)
%   2. lon = longitude (positive for eastern hemisphere, negative for western)
%   3. t_zone = indicates the time zone correction factor (East = 5, Central = 6,
%   Mountain = 7, Pacific = 8)
%   4. z = site elevation above sea level (m)
%
% Output:
% temp_mod      - modeled stream temperatures through space and time (deg C)
% matrix_data   - a structure containing the intermediate steps used
%                 to calculate the finite difference solution.
%                 Includes a,b,c,A,o,p,q,g,k,m, and d
% node_data     - a structure containing the volume of each node (volume, m3),
%                 the interpolated discharge measurements at each
%                 node (Q, m3/min), and the lateral inflow calculated at each
%                 node (ql, m3/min)
% flux_data     - a structure containing the total heat flux (heatflux),
%                 solar radiation (solarflux), atmospheric longwave
%                 radiation (atmflux), land cover radiation (landflux),
%                 back radiation from the stream (backrad), latent
%                 heat flux (evap),stream bed conduction
%                 (conduction), and sensible heat flux (sens), all
%                 in W/m2, at each node and time_step

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 12/4/2016
%% Select solution method (1=Crank-Nicolson, 2=second order Runge-Kutta):
method=input_data.settings(1,1);
if method==1
    compmeth='Solution Method=Crank-Nicolson';
else
    compmeth='Solution Method=Runge-Kutta';
end
%% set up switch to turn off displays and user inputs
unattend=input_data.settings(5,1);

%% Assign appropriate variable names to columns of input data
if ~unattend
    disp(' ')
    disp(compmeth)
    disp(' ')
    disp('Assigning variable names...')
end

time_mod=input_data.time_mod;
dist_mod=input_data.dist_mod;
time_temp=input_data.temp_x0_data(:,1);
temp_x0=input_data.temp_x0_data(:,2);
dist=input_data.temp_t0_data(:,1);
temp_t0=input_data.temp_t0_data(:,2);
dist_stdim=input_data.dim_data(:,1);
% area=input_data.dim_data(:,2);
width=input_data.dim_data(:,3);
depth=input_data.dim_data(:,4);
discharge_stdim = input_data.dim_data(:,5); 
dist_dis=input_data.dis_data(:,1);
discharge=input_data.dis_data(:,2:end); %allow for multiple columns
time_dis=input_data.time_dis(:,1); % the times that the discharge was measured in minutes
dist_T_L=input_data.T_L_data(:,1);
T_L=input_data.T_L_data(:,2);
year=input_data.met_data(:,1);
month=input_data.met_data(:,2);
day=input_data.met_data(:,3);
hour=input_data.met_data(:,4);
minute=input_data.met_data(:,5);
time_met=input_data.met_data(:,6);
solar_rad_in=input_data.met_data(:,7);
air_temp_in=input_data.met_data(:,8);
rel_hum_in=input_data.met_data(:,9);
wind_speed_in=input_data.met_data(:,10);
dist_bed=input_data.bed_data1(:,1);
depth_of_meas=input_data.bed_data1(:,2);
time_bed=input_data.bed_data2(1,:);
bed_temp=input_data.bed_data2(2:end,:);
sed_type=input_data.sed_type;
dist_shade=input_data.shade_data(:,1);
shade=input_data.shade_data(:,2);
vts=input_data.shade_data(:,3);
time_cloud=input_data.cloud_data(:,1);
c_in=input_data.cloud_data(:,2);
lat=input_data.site_info(1,1);
lon=input_data.site_info(2,1);
t_zone=input_data.site_info(3,1);
z=input_data.site_info(4,1);
% Interpolate bed data
[sed]=hflux_bed_sed(sed_type,dist_bed,dist_mod);
% Calculate reflected shortwave radiation
[sol_refl]=hflux_shortwave_refl(year,month,day,hour,...
    minute,lat,lon,t_zone,time_met,time_mod);

if ~unattend
    disp('...done!')
    disp('  ')
end

%%  Determine the number of time steps for the model run and the number of
%  reservoirs, or nodes, along the stream.  Also determine time
%  step, dt.

if ~unattend
    disp('Determining time steps and nodes...')
end

timesteps=numel(time_mod);
r=numel(dist_mod);
dt=max(time_mod)/(numel(time_mod)-1); %delta time


if ~unattend
    disp('...done!')
    disp('  ')
end


%% Interpolate all input data given longitudinally so that there are values
% at every node


if ~unattend
    disp('Interpolating longitudinal data in space...')
end

T_L_m(:,1)=interp1(dist_T_L(:,1),T_L(:,1),dist_mod(:,1));
temp_t0_m(:,1)=interp1(dist(:,1),temp_t0(:,1),dist_mod(:,1));
discharge_m(:,:)=interp1(dist_dis(:,1),discharge(:,:),dist_mod(:,1)); % made the code interpolate in space multiple discharge columns
width_m(:,1)=interp1(dist_stdim(:,1),width(:,1),dist_mod(:,1));
depth_m(:,1)=interp1(dist_stdim(:,1),depth(:,1),dist_mod(:,1));
depth_of_meas_m(:,1)=interp1(dist_bed(:,1),depth_of_meas(:,1),dist_mod(:,1));
shade_m(:,1)=interp1(dist_shade(:,1),shade(:,1),dist_mod(:,1));
vts_m(:,1)=interp1(dist_shade(:,1),vts(:,1),dist_mod(:,1));
for n=1:length(time_bed)
    bed_temp_m(:,n)=interp1(dist_bed(:,1),bed_temp(:,n),dist_mod(:,1));
end

if ~unattend
    disp('...done!')
    disp('  ')
end


%% Interpolate all data given through time so that there are
%  values at every time step

if ~unattend
    disp('Interpolating temporal data in time...')
end

discharge_t = discharge_m'; %transpose discharge matrix that was interpolated in space
discharge_m=interp1(time_dis,discharge_t,time_mod); % interpolate discharge through time
discharge_m = discharge_m'; %transform back so each row is a location along the stream and each column is a different time

% Calculate width-depth-discharge relationship  
theta = zeros(r,1); %assumes stream cross section is a triangle and calculates the angle between the depth and the edge
for n = 1:r
    theta(n,1) = atan((0.5*width_m(n,1))/depth_m(n,1));
end

dim_Q = interp1(dist_stdim(:,1),discharge_stdim(:,1),dist_mod(:,1)); %linear interpolation of stream discharge at each location where dimensions were measured to each model node
n_s = zeros(r,1); % calculates the manning number divided by the slope^1/2 which will be used as a constant at each location along the stream
for n = 1:r
    n_s(n,1) = ((0.25^(2/3))*(2^(5/3))*(cos(theta(n,1))^(2/3))*(tan(theta(n,1)).^(5/3))*(depth_m(n,1)^(8/3)))/(2*dim_Q(n,1));
end

% Solve for depth through time and distance (assume constant theta and n_s)
depth_m = zeros(r,timesteps); 
for n = 1:r
    for t = 1:timesteps
        depth_m(n,t) = ((2*n_s(n,1)*discharge_m(n,t))/((0.25^(2/3))*(2^(5/3))*(cos(theta(n,1))^(2/3))*(tan(theta(n,1))^(5/3)))).^(3/8);
    end
end

% Solve for width of stream through time and distance (assume constant theta and n_s)
width_m = zeros(r,timesteps);
for n = 1:r
    for t = 1:timesteps
        width_m(n,t) = 2*tan(theta(n,1))*(((2*n_s(n,1)*discharge_m(n,t))/((0.25^(2/3))*(2^(5/3))*(cos(theta(n,1))^(2/3))*(tan(theta(n,1))^(5/3)))).^(3/8));
    end
end

area_m = 0.5*depth_m.*width_m; % assumes triangular cross sectional area

% Calculate wetted perimeter through time and space for triangular cross section
WP_m = zeros(r,timesteps);
for t = 1:timesteps
    WP_m(:,t) = 2*(depth_m(:,t)./cos(theta(:,1)));
end  

solar_rad_dt(:,1)=interp1(time_met(:,1),solar_rad_in(:,1),time_mod(:,1));
air_temp_dt(:,1)=interp1(time_met(:,1),air_temp_in(:,1),time_mod(:,1));
rel_hum_dt(:,1)=interp1(time_met(:,1),rel_hum_in(:,1),time_mod(:,1));
wind_speed_dt(:,1)=interp1(time_met(:,1),wind_speed_in(:,1),time_mod(:,1));
c_dt(:,1)=interp1(time_cloud(:,1),c_in(:,1),time_mod(:,1));
temp_x0_dt=interp1(time_temp,temp_x0,time_mod,'pchip');
temp_x0_dt=temp_x0_dt';
for n=1:r
    bed_temp_dt(n,:)=interp1(time_bed(1,:),bed_temp_m(n,:),time_mod(:,1));
end

%% Create distance x time matrices from the met data (assume met data is
%  constant in space

% pre-allocate met data matrices
solar_rad_mat=zeros(r,timesteps);  %pre-allocate solar radiation array
air_temp_mat=zeros(r,timesteps);  %pre-allocate air temperature array
rel_hum_mat=zeros(r,timesteps);  %pre-allocate relative humidity array
wind_speed_mat=zeros(r,timesteps);  %pre-allocate wind speed array
cl=zeros(r,timesteps);  %pre-allocate cloud cover array

for n=1:r
    solar_rad_mat(n,:)=solar_rad_dt';
    air_temp_mat(n,:)=air_temp_dt';
    rel_hum_mat(n,:)=rel_hum_dt';
    wind_speed_mat(n,:)=wind_speed_dt';
    cl(n,:)=c_dt';
end

if ~unattend
    disp('...done!')
    disp('  ')
end

%% STEP 1: compute volumes of each reservoir (node) in the model

if ~unattend
    disp('Computing volumes of nodes, discharge rates and groundwater inflow rates...')
end

volume(r,timesteps)=0; %pre-allocate volume array
volume(1,:)=(dist_mod(2,1)-dist_mod(1,1))*area_m(1,:); %set volume at first node
for n=2:r-1   %set volumes at nodes not at the boundaries
    volume(n,:)=area_m(n,:)*(((dist_mod(n+1,1)+dist_mod(n,1))/2)-((dist_mod(n,1)...
        +dist_mod(n-1,1))/2));
end
volume(r,:)=area_m(r,:)*(dist_mod(r,1)-dist_mod(r-1,1));  %set volume at last node


%% STEP 2: compute discharge rates at reservoir edges using linear
%  interpolation (n-values are upstream of node n values)

Q_half(r+1,timesteps)=0; %pre-allocate array; originally Q_half(r+1,1)=0;
% extrapolate backwards to get discharge rate at upstream edge of stream
% (half a node upstream from first node)
Q_half(1,:)=(((2*discharge_m(1,:))-discharge_m(2,:))+discharge_m(1,:))./2;
% Use the following line to set discharge at upstream edge equal to the discharge at
% the first node, rather than extrapolating (an alternative to line 47).
% Q_half(1,:)=discharge(1,:);
for n=2:r  % linearly interpolate to get discharge rates at node boundaries
    Q_half(n,:)=(discharge_m(n,:)+discharge_m(n-1,:))./2; % originally Q_half(n,1)=(discharge_m(n,1)+discharge_m(n-1,1))/2;
end
%extrapolate forwards to get discharge rate at downstream edge of stream
%(half a node downstream from last node)
Q_half(r+1,:)=(((2*discharge_m(r,:))-discharge_m(r-1,:))+discharge_m(r,:))./2; % originally Q_half(r+1,1)=(((2*discharge_m(r,1))-discharge_m(r-1,1))+discharge_m(r,1))/2;
% Use the following line to set discharge at downstream edge equal to the 
% discharge at the last node, rather than extrapolating 
% (an alternative to line 56).
% Q_half(r+1,:)=discharge(r,:);


% STEP 3: compute lateral groundwater discharge rates to each node based on
% longtudinal changes in streamflow

Q_L(r,timesteps)=0; %pre-allocate array: originally Q_L(r,1)=0;
for n=1:r
    Q_L(n,:)=Q_half(n+1,:)-Q_half(n,:); %originally  Q_L(n,1)=Q_half(n+1)-Q_half(n);
end

% Compute lateral groundwater discharge rates to each node based on changes
% in streamflow through time (m^3/s) ==NEW

% Q_L_t(size(Q_half,1),timesteps-1)=zeros; %pre-allocate array
% for n=1:timesteps-1
%     Q_L_t(:,n)=Q_half(:,n+1)-Q_half(:,n);
% end
%% STEP 4: unit conversions so all discharge rates are in m3/min;
%  note that all inputs are x are in m, T are in deg C, and Q or Q_L are in m3/s

Q_half_min=Q_half(1:(r+1),:)*60; %Originally Q_half_min=Q_half(1:(r+1),1)*60;
Q_L_min=Q_L(1:r,:)*60; %Originally Q_L_min=Q_L(1:r,1)*60;
% Q_L_t_min=Q_L_t*60; %NEW to compute change in discharge through time in m3/min

if ~unattend    
    disp('...done!')
    disp('  ')
end

switch method
    case 1
        %% STEP 5: Calculate coefficients of the tridiagonal matrix (a, b, c)
        % and set coefficients at the boundaries. Use a, b and c to create the A
        % matrix.  Note that a, b and c are constant in time as long as Q,
        % volume, and Q_L are constant with time.
        
        if ~unattend
            disp('Computing A matrix coefficients...')
        end
        
        %preallocate coefficient matrices
        a=zeros(r,timesteps);
        b=zeros(r,timesteps);
        c=zeros(r,timesteps);
                
        for n=1:r %pieces of equation for the future temperature
            for i = 1:timesteps-1 %added for loop for discharge through time (what to do about last time?)
            a(n,i) = (-dt*Q_half_min(n,i+1))/(4*volume(n,1)); %subdiagonal: coefficients of Tn-1
            o(n,i)=(dt*Q_half_min(n,i+1))/((4*volume(n,1))); %intermediate steps to get b
            p(n,i)=(dt*Q_half_min(n+1,i+1))/((4*volume(n,1)));
            q(n,i)=(dt*Q_L_min(n,i+1))/((2*volume(n,1)));
            b(n,i) = 1+o(n,i)-p(n,i)+q(n,i); % diagonal: coefficients of Tn
            c(n,i) = (dt*Q_half_min(n+1,i+1))/(4*volume(n,1)); %superdiagonal: coefficients of Tn+1
            end
            %for last values since we don't know the future we just reuse
            %the last known
            a(n,timesteps) = (-dt*Q_half_min(n,timesteps))/(4*volume(n,1)); %subdiagonal: coefficients of Tn-1
            o(n,timesteps)=(dt*Q_half_min(n,timesteps))/((4*volume(n,1))); %intermediate steps to get b
            p(n,timesteps)=(dt*Q_half_min(n+1,timesteps))/((4*volume(n,1)));
            q(n,timesteps)=(dt*Q_L_min(n,timesteps))/((2*volume(n,1)));
            b(n,timesteps) = 1+o(n,timesteps)-p(n,timesteps)+q(n,timesteps); % diagonal: coefficients of Tn
            c(n,timesteps) = (dt*Q_half_min(n+1,timesteps))/(4*volume(n,1));
        end
        
        
        for n=1:r %pieces of equation for the current temperature to make d (new)
            for i = 1:timesteps 
            a_c(n,i) = (-dt*Q_half_min(n,i))/(4*volume(n,1)); %subdiagonal: coefficients of Tn-1
            o_c(n,i)=(dt*Q_half_min(n,i))/((4*volume(n,1))); %intermediate steps to get b
            p_c(n,i)=(dt*Q_half_min(n+1,i))/((4*volume(n,1)));
            q_c(n,i)=(dt*Q_L_min(n,i))/((2*volume(n,1)));
            b_c(n,i) = 1+o_c(n,i)-p_c(n,i)+q_c(n,i); % diagonal: coefficients of Tn
            c_c(n,i) = (dt*Q_half_min(n+1,i))/(4*volume(n,1)); %superdiagonal: coefficients of Tn+1
            end
        end
        
        if ~unattend
            disp('...done!')
            disp('  ')
        end
        
        %% STEP 6: Calculate right hand side (d).
        % The values for d are temperature-dependent, so they change each time step.
        % Once d is computed, use that d value and the
        % matrix A to solve for the temperature for each time step.
        
        if ~unattend
            disp('Computing d-values, heat fluxes and solving for stream temperatures...')
        end
        
        d=zeros(r,timesteps);  %pre-allocate d array
        T=zeros(r,timesteps);  %pre-allocate temperature array
        T(:,1)=temp_t0_m;  %set temperatures for first time step at input values
        
        % Initialize the heat flux matrix
        heat_flux=zeros(r,timesteps);
        % Insert R-values for first time step.
        [heat_flux(:,1) shortwave(:,1) longwave(:,1) atm(:,1) back(:,1) land(:,1) ...
            latent(:,1) sensible(:,1) bed(:,1)]=hflux_flux(input_data.settings,solar_rad_mat(1:r,1),...
            air_temp_mat(1:r,1),rel_hum_mat(1:r,1),temp_t0_m(1:r,1),...
            wind_speed_mat(1:r,1),z,sed(1:r,1),bed_temp_dt(1:r,1),...
            depth_of_meas_m(1:r,1),shade_m(1:r,1),vts_m(1:r,1),cl(1:r,1),sol_refl(1,1),WP_m(1:r,1),width_m(1:r,1));
        
        % Assign variables to calculate R term
        
        % Density of water
        rho_water=1000; %(kg/m^3)
        
        % Specific heat capacity of water
        c_water=4182; %(J /kg deg C)
        
        
        % Pre-allocate g, k, and m arrays
        g(r,timesteps)=zeros;
        k(r,timesteps)=zeros;
        m(r,timesteps)=zeros;
        
        
        if ~unattend
            h = waitbar(0,'Calculating...');
        end
        
        g=1+a_c+c_c-q_c; %originally g = 1+a+c-q';
        for n=1:r
            for i = 1:timesteps
                if Q_L_min(n,i)<0
                k(n,i)=0;
                else
                k(n,i)=(dt*Q_L_min(n,i))/(volume(n,1));
                end
            end
        end        
        
        for t=1:timesteps
            if t==timesteps
            else
                m(:,t)=(dt.*(width_m(:,t).*heat_flux(:,t))./((rho_water*c_water))./area_m(:,t));
                d(1,t)= ((g(1,t)*T(1,t))+(o_c(1,t)*temp_x0_dt(1,t))-(p_c(1,t)...
                            *T(1+1,t))+(k(1,t)*T_L_m(1,1))+m(1,t) ) - (a_c(1,t)*temp_x0_dt(1,t+1)); %at upper boundary of reach
                d(r,t)=(g(r,t)*T(r,t))+(o_c(r,t)*T(r-1,t))-(p_c(r,t)*T(r,t))+...
                            (k(r,t)*T_L_m(r,1))+m(r,t); %at lower boundary of reach
                d(2:r-1,t)=(g(2:r-1,t).*T(2:r-1,t))+(o_c(2:r-1,t).*T(1:r-2,t))-(p_c(2:r-1,t).*T(3:r,t))...
                            +(k(2:r-1,t).*T_L_m(2:r-1,1))+m(2:r-1,t); %untransformed some of the parts of this equation
                
                A = zeros(r,r);
                for n=1:r-1;
                    A(n+1,n)=a(n+1,t);
                    A(n,n)=b(n,t);
                    A(n,n+1)=c(n,t);
                    %  A(r,r)=b(r);  
                    A(r,r)=b(r,t)+c(r,t);  %This creates a specified temperature gradient of
            % dT/dx=0 at the downtream node.  In other words, T(r)=T(r+1),
            % with T(r+1) being a dummy cell after the model reach.
                end
                T(1:r,t+1)=A\d(:,t);
                
                % Insert R-values for current time step
                [heat_flux(:,t+1), shortwave(:,t+1) longwave(:,t+1)...
                    atm(:,t+1) back(:,t+1) land(:,t+1) latent(:,t+1)...
                    sensible(:,t+1) bed(:,t+1)]...
                    =hflux_flux(input_data.settings,solar_rad_mat(:,t+1),air_temp_mat(:,t+1),...
                    rel_hum_mat(:,t+1),T(:,t+1),wind_speed_mat(:,t+1),z,...
                    sed(:,1),bed_temp_dt(:,t+1),depth_of_meas_m(:,1),...
                    shade_m(:,1),vts_m(:,1),cl(:,t+1),sol_refl(t+1,1),WP_m(:,t+1),width_m(:,t+1));
            end
            if ~unattend
                waitbar(t/timesteps,h);
            end
        end
        
        if ~unattend
            close(h);
            disp('...done!')
            disp('  ')
        end
        
        
    case 2
        if ~unattend
            disp('Computing heat fluxes and solving for stream temperatures...')
        end
        %% STEP 6: Pre-allocate arrays used in iterative solution and set
        % initial and boundary temperature conditions
        
        T=zeros(r,timesteps);  %pre-allocate temperature array
        T(:,1)=temp_t0_m;  %set initial temperature condition
        T(1,:)=temp_x0_dt;  %set boundary temperature condition
        % pre-allocate arrays for temperature at intermediate times
        T_k1=zeros(r,timesteps);
        % Initialize the heat flux matrix for the sink/source term
        heat_flux=zeros(r,timesteps);
        % Insert R-values for first time step.
        [heat_flux(:,1), shortwave(:,1), longwave(:,1), atm(:,1), back(:,1),...
            land(:,1), latent(:,1), sensible(:,1), bed(:,1)] = hflux_flux(...
            input_data.settings,solar_rad_mat(1:r,1), air_temp_mat(1:r,1), rel_hum_mat(1:r,1),...
            temp_t0_m(1:r,1), wind_speed_mat(1:r,1), z, sed(1:r,1),...
            bed_temp_dt(1:r,1), depth_of_meas_m(1:r,1), shade_m(1:r,1),...
            vts_m(1:r,1), cl(1:r,1), sol_refl(1,1),WP_m(:,1),width_m(:,1));
        %% Step 7: Calculate stream temperature using a second order Runge-Kutta method.
        % Change in temperature with respect to time is
        % calculated at intermediate timesteps as defined by k1, which is
        % used to determine temperature at the intermediate time step.  k1
        % is then used to calculate change in temperature to the next time
        % step, or k2.  Temperature at the future timestep is solved using
        % k1 and k2.
        if ~unattend
            h = waitbar(0,'Calculating...');
        end
        for t=1:timesteps
            if t==timesteps
            else
                % Insert heat flux for k1 at n=2:r-1
                [heat_flux_k1(:,t), ~, ~, ~, ~, ~, ~, ~, ~,]...
                    =hflux_flux(input_data.settings,solar_rad_mat(:,t),air_temp_mat(:,t),...
                    rel_hum_mat(:,t),T(:,t),wind_speed_mat(:,t),z,...
                    sed(:,1),bed_temp_dt(:,t),depth_of_meas_m(:,1),...
                    shade_m(:,1),vts_m(:,1),cl(:,t),sol_refl(t,1),WP_m(:,t),width_m(:,t));
                for n=2:r-1
                    u1(n,t)=(Q_half_min(n,t)/volume(n,1))*(0.5*T(n-1,t)-0.5*T(n,t));
                    v1(n,t)=(Q_half_min(n+1,t)/volume(n,1))*(0.5*T(n,t)-0.5*T(n+1,t));
                    s1(n,t)=(Q_L_min(n,t)/volume(n,1))*(T_L_m(n,1)-T(n,t));
                    % Density of water
                    rho_water=1000; %(kg/m^3)
                    % Specific heat capacity of water
                    c_water=4182; %(J /kg deg C)
                    m1(n,t)=(width_m(n,t).*heat_flux_k1(n,t))/...
                        ((rho_water*c_water))/area_m(n,t);
                    k1(n,t)=u1(n,t)+v1(n,t)+s1(n,t)+m1(n,t);
                end
                % T(r,t) is assumed to be T(r-1,t)
                for n=r
                    u1(n,t)=(Q_half_min(n,t)/volume(n,1))*(0.5*T(n-1,t)-0.5*T(n,t));
                    v1(n,t)=(Q_half_min(n+1,t)/volume(n,1))*(0.5*T(n,t)-0.5*T(n,t));
                    s1(n,t)=(Q_L_min(n,t)/volume(n,1))*(T_L_m(n,1)-T(n,t));
                    m1(n,t)=(width_m(n,t).*heat_flux_k1(n,t))/...
                        ((rho_water*c_water))/area_m(n,t);
                    k1(n,1)=u1(n,t)+v1(n,t)+s1(n,t)+m1(n,t);
                end
                % Calculate temperature based on k1
                T_k1(1,t)=temp_x0_dt(1,t);  %set boundary temperature condition
                for n=2:r
                    T_k1(n,t)=T(n,t)+(dt*k1(n,t));
                end
                % Calculate k2
                % Insert R-values for k2
                [heat_flux_k2(:,t), ~, ~, ~, ~, ~, ~, ~, ~,]...
                    =hflux_flux(input_data.settings,solar_rad_mat(:,t),air_temp_mat(:,t),...
                    rel_hum_mat(:,t),T_k1(:,t),wind_speed_mat(:,t),z,...
                    sed(:,1),bed_temp_dt(:,t),depth_of_meas_m(:,1),...
                    shade_m(:,1),vts_m(:,1),cl(:,t),sol_refl(t,1),WP_m(:,t),width_m(:,t));
                for n=2:r-1
                    u2(n,t)=(Q_half_min(n,t+1)/volume(n,1))*(0.5*T_k1(n-1,t)-0.5*T_k1(n,t));
                    v2(n,t)=(Q_half_min(n+1,t+1)/volume(n,1))*(0.5*T_k1(n,t)-0.5*T_k1(n+1,t));
                    s2(n,t)=(Q_L_min(n,t+1)/volume(n,1))*(T_L_m(n,1)-T_k1(n,t));
                    m2(n,t)=(width_m(n,t).*heat_flux_k2(n,t))/...
                        ((rho_water*c_water))/area_m(n,t);
                    k2(n,t)=u2(n,t)+v2(n,t)+s2(n,t)+m2(n,t);
                end
                % T(r,t) is assumed to be T(r-1,t)
                for n=r
                    u2(n,t)=(Q_half_min(n,t+1)/volume(n,1))*(0.5*T_k1(n-1,t)-0.5*T_k1(n,t));
                    v2(n,t)=(Q_half_min(n+1,t+1)/volume(n,1))*(0.5*T_k1(n,t)-0.5*T_k1(n,t));
                    s2(n,t)=(Q_L_min(n,t+1)/volume(n,1))*(T_L_m(n,1)-T_k1(n,t));
                    m2(n,t)=(width_m(n,t).*heat_flux_k2(n,t))/...
                        ((rho_water*c_water))/area_m(n,t);
                    k2(n,t)=u2(n,t)+v2(n,t)+s2(n,t)+m2(n,t);
                end
                % Calculate temperature at next timestep for each node
                for n=2:r
                    T(n,t+1)=T(n,t)+(dt*(0.5*k1(n,t)+0.5*k2(n,t)));
                end
                % Insert R-values for future time step
                [heat_flux(:,t+1), shortwave(:,t+1), longwave(:,t+1),...
                    atm(:,t+1), back(:,t+1), land(:,t+1), latent(:,t+1),...
                    sensible(:,t+1), bed(:,t+1)]...
                    =hflux_flux(input_data.settings,solar_rad_mat(:,t+1),air_temp_mat(:,t+1),...
                    rel_hum_mat(:,t+1),T(:,t+1),wind_speed_mat(:,t+1),z,...
                    sed(:,1),bed_temp_dt(:,t+1),depth_of_meas_m(:,1),...
                    shade_m(:,1),vts_m(:,1),cl(:,t+1),sol_refl(t+1,1),WP_m(:,t+1),width_m(:,t+1));
            end
            if ~unattend
                waitbar(t/timesteps,h);
            end
        end
        if ~unattend
            close(h);
        end
        
        if ~unattend
            disp('...done!')
            disp('  ')
        end
end
%% Make plots
if ~unattend
    
    % 2D plot of stream temperature
    f0 = figure('Visible','on');
    imagesc(time_mod,dist_mod,T);
    colormap(jet(256));
    t=colorbar;% Inserts a colorbar. a handle is created
    set(get(t,'ylabel'),'string',['Temperature (' char(176) 'C)'],...
        'FontSize',12,'FontWeight','Bold')
    axis square;
    plot_title='Modeled Stream Temperature';
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
    
    % 3D plot of stream temperature
    f1 = figure('Visible','on');
    y=dist_mod;
    surf(time_mod,y,T);
    colormap(jet(256));
    t=colorbar;% Inserts a colorbar. a handle is created
    set(get(t,'ylabel'),'string',['Temperature (' char(176) 'C)'],...
    'FontSize',11,'FontWeight','Bold')
    shading 'flat'
    axis square;
    set(gca,'FontName','Arial');
    plot_title='Modeled Stream Temperature';
    font_type='Arial';
    xlab='Time (min)';
    ylab='Distance (m)';
    zlab=['Temp (' char(176) 'C)'];
    title(plot_title,...
        'FontName' , font_type, ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    xlabel(xlab,...
        'FontName' , font_type, ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    ylabel(ylab,...
        'FontName' , font_type, ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    zlabel(zlab,...
        'FontName' , font_type, ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    
    % Plot of heat fluxes
    f2=figure;
    subplot(3,2,1);
    plot(time_mod,mean((heat_flux/60)),'k');
    set(gca,'FontName','Arial');
    title('Total Heat Flux','FontWeight','Bold');
    ylabel('Energy Flux (W/m^2)','FontWeight','Bold')
    axis([min(time_mod) max(time_mod) min(mean((heat_flux/60))) max(mean((heat_flux/60)))])
    subplot(3,2,2);
    plot(time_mod,mean(shortwave),'r');
    title('Shortwave Radiation','FontWeight','Bold');
    axis([min(time_mod) max(time_mod) min(mean(shortwave)) max(mean(shortwave))])
    subplot(3,2,3);
    plot(time_mod,mean(longwave),'b');
    title('Longwave Radiation','FontWeight','Bold');
    ylabel('Energy Flux (W/m^2)','FontWeight','Bold')
    axis([min(time_mod) max(time_mod) min(mean(longwave)) max(mean(longwave))])
    subplot(3,2,4);
    plot(time_mod,mean(latent),'g');
    title('Latent Heat Flux','FontWeight','Bold');
    axis([min(time_mod) max(time_mod) min(mean(latent)) max(mean(latent))])
    subplot(3,2,5);
    plot(time_mod, mean(bed),'m');
    title('Bed Conduction','FontWeight','Bold');
    xlabel('Time (min)','FontWeight','Bold')
    ylabel('Energy Flux (W/m^2)','FontWeight','Bold')
    axis([min(time_mod) max(time_mod) min(mean(bed)) max(mean(bed))])
    subplot(3,2,6);
    plot(time_mod, mean(sensible),'y');
    title('Sensible Heat Flux','FontWeight','Bold');
    xlabel('Time (min)','FontWeight','Bold')
    axis([min(time_mod) max(time_mod) min(mean(sensible)) max(mean(sensible))])
    
    % Plot of heat fluxes: comparison
    f3=figure;
    plot(time_mod,(mean(heat_flux)/60),'k');
    hold on
    plot(time_mod,mean(shortwave),'r');
    plot(time_mod,mean(longwave),'b');
    plot(time_mod,mean(latent),'g');
    plot(time_mod,mean(bed),'c');
    plot(time_mod,mean(sensible),'m');
    set(gca,'FontName','Arial');
    axis normal
    set(gca,'XLim',[min(time_mod) max(time_mod)])
    title('Energy Fluxes',...
        'FontName' , 'Arial', ...
        'FontSize'   , 12 ,'FontWeight','Bold' );
    xlabel('Time (min)',...
        'FontName' , 'Arial', ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    ylabel('Energy Flux (W/m^2)',...
        'FontName' , 'Arial', ...
        'FontSize'   , 11 ,'FontWeight','Bold' );
    legend('Total Heat Flux','Solar Radiation','Longwave Radiation',...
        'Latent Heat Flux','Streambed Conduction','Sensible Heat Flux','Location','Best');
    
end

%% Write results to output
temp_mod=T;
if method==1
    matrix_data.a=a;
    matrix_data.b=b;
    matrix_data.c=c;
    matrix_data.A=A;
    matrix_data.o=o;
    matrix_data.p=p;
    matrix_data.q=q;
    matrix_data.g=g;
    matrix_data.k=k;
    matrix_data.m=m;
    matrix_data.d=d;
    
    matrix_data.a_c=a_c; %new
    matrix_data.b_c=b_c;
    matrix_data.c_c=c_c;
    matrix_data.o_c=o_c;
    matrix_data.p_c=p_c;
    matrix_data.q_c=q_c;
else
    matrix_data.u1=u1;
    matrix_data.v1=v1;
    matrix_data.s1=s1;
    matrix_data.m1=m1;
    matrix_data.k1=k1;
    matrix_data.u2=u2;
    matrix_data.v2=v2;
    matrix_data.s2=s2;
    matrix_data.m2=m2;
    matrix_data.k2=k2;
end
node_data.v=volume;
node_data.Q=Q_half_min;
node_data.ql=Q_L_min;
% node_data.qlt=Q_L_t_min; %new
node_data.width=width_m;
node_data.area=area_m;
flux_data.heatflux=heat_flux/60;
flux_data.solarflux=shortwave;
flux_data.solar_refl=sol_refl;
flux_data.long=longwave;
flux_data.atmflux=atm;
flux_data.landflux=land;
flux_data.backrad=back;
flux_data.evap=latent;
flux_data.sensible=sensible;
flux_data.conduction=bed;
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