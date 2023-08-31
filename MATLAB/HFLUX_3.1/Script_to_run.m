%% Script to Run HFLUX
% First runs HFLUX, then determine error on calculated temperature, lastly
% run simple sensitivity analysis
[input_data] = hflux_format('example_data.xlsx');     
[temp_mod matrix_data node_data flux_data] = hflux(input_data);
temp=xlsread('example_data.xlsx','temp');
[rel_err me mae mse rmse nrmse]=hflux_error(input_data.time_mod,input_data.dist_mod,temp_mod,input_data.temp_t0_data(:,1),input_data.temp_x0_data(:,1),temp);
[sens] = hflux_sens(input_data,[-0.01 0.01],[-2 2],[-0.1 0.1],[-0.1 0.1])

%% Save outputs external from Matlab
writematrix(temp_mod, 'temp_mod.csv') % save output to csv file
writematrix(temp, 'temp.csv') 
writematrix(rel_err, 'rel_err.csv')

writematrix(flux_data.heatflux, "heatflux_data.csv")
writematrix(flux_data.solarflux, "solarflux_data.csv")
writematrix(flux_data.solar_refl, "solar_refl_data.csv")
writematrix(flux_data.long, "long_data.csv")
writematrix(flux_data.atmflux, "atmflux_data.csv")
writematrix(flux_data.landflux, "landflux_data.csv")
writematrix(flux_data.backrad, "backrad_data.csv")
writematrix(flux_data.evap, "evap_data.csv")
writematrix(flux_data.sensible, "sensible_data.csv")
writematrix(flux_data.conduction, "conduction_data.csv")