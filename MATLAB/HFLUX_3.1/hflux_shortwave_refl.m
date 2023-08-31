%% REFLECTED SHORTWAVE RADIATION CALCULATIONS

% Calculates the reflected shortwave radiation from the stream surface
% using the solar calculations and Fresnel's reflectance.
% See Boyd and Kasper, 2003 for a description of the solar equations

% Input:
%   Note: must all be for the same time period and distance (ie same size)
%   year = year during which measurements were taken (ie. 2012)
%   month = month during which measurements were taken (ie. 6)
%   day = day during which measurements were taken (ie. 18)
%   hour = hour during which measurements were taken (ie. 22) (military
%      time)
%   min = minute during which measurements were taken (ie. 5)
%   time_met = times at which meteorological data are known, in minutes (met
%       data includes air temp, rel humidity, wind speed, and solar rad)
%   lat = latitude (positive for northern hemisphere, negative for southern)
%   lon = longitude (positive for eastern hemisphere, negative for western)
%   t_zone = indicates the time zone correction factor (East = 5, Central = 6,
%       Mountain = 7, Pacific = 8)
%   time_mod = model times at which temperatures will be computed for heat budget,
%   in minutes
%
% Output:
%  [sol_refl] = portion of solar radiation reflected off the surface of the
%  stream

% Written by AnneMarie Glose, Syracuse University, April 2013
%   Department of Earth Sciences
%   204 Heroy Geology Lab
%   Syracuse, NY  13244
%   Contact: amglose@syr.edu
% Copyright (c) 2013, AnneMarie Glose. All rights reserved.
% Last updated 10/28/2014

function [sol_refl]=hflux_shortwave_refl(year,month,day,hour,minute,lat,lon,t_zone,time_met,time_mod)

for n=1:length(year)
%  a. Solar time relative to Earth 
    time_frac(n,1)=(hour(n,1)*(1/24))+(minute(n,1)*(1/60)*(1/24));
    % Determine if daylight savings correction is necessary
    if (year(n,1)<2007)
        if (month(n,1) > 4) && (month(n,1) < 10) %If the month is between April and October, true
            dst = true;
        elseif (month(n,1) < 3) || (month(n,1) == 12) %If month is January, February, or December, false
            dst = false;
        elseif month(n,1) == 4 %If month is April
            sunday = [];
            for i=1:30
                DateNumber = datenum(year(n,1), month(n,1), i);
                DOW = weekday(DateNumber);
                if DOW == 1 %DOW is 1 if the day is a Sunday
                    sunday = [sunday, i];
                end
            end
            DS_Day = sunday(1); %Take the 21st Sunday in April
            if day(n,1) >= DS_Day %If on or after 1st Sunday, true
                dst(n,1) = true;
            else
                dst(n,1) = false;
            end
        elseif month(n,1) == 10 %If month is October
            sunday = [];
            for i=1:31
                DateNumber = datenum(year, month, i);
                DOW = weekday(DateNumber);
                if DOW == 1 %DOW is 1 if the day is a Sunday
                    sunday = [sunday, i];
                end
            end
            DS_Day = sunday(end); %Take the last Sunday in October
            if day(n,1) >= DS_Day %If on or after the last Sunday, true
                dst(n,1) = true;
            else
                dst(n,1) = false;
            end
        end
    else
        if (month(n,1) > 3) && (month(n,1) < 11) %If the month is between April and October, true
            dst(n,1) = true;
        elseif (month(n,1) < 3) || (month(n,1) == 12) %If month is January, February, or December, false
            dst(n,1) = false;
        elseif month(n,1) == 3 %If month is March
            sunday = [];
            for i=1:31
                DateNumber = datenum(year, month, i);
                DOW = weekday(DateNumber);
                if DOW == 1 %DOW is 1 if the day is a Sunday
                    sunday = [sunday, i];
                end
            end
            DS_Day = sunday(2); %Take the 2nd Sunday in March
            if day(n,1) >= DS_Day %If on or after 2nd Sunday, true
                dst(n,1) = true;
            else
                dst(n,1) = false;
            end
        elseif month(n,1) == 11 %If month is November
            sunday = [];
            for i=1:30
                DateNumber = datenum(year, month, i);
                DOW = weekday(DateNumber);
                if DOW == 1 %DOW is 1 if the day is a Sunday
                    sunday = [sunday, i];
                end
            end
            DS_Day = sunday(1); %Take the 1st Sunday in November
            if day(n,1) >= DS_Day %If on or after the 1st Sunday, true
                dst(n,1) = true;
            else
                dst(n,1) = false;
            end
        end
    end
    if dst(n,1)==true
        t_dst(n,1)=time_frac(n,1)+1/24;
    else
        t_dst(n,1)=time_frac(n,1);
    end
    t_gmt(n,1)=t_dst(n,1)+(t_zone/24);
    A1(n,1)=round(year(n,1)/100);
    B1(n,1)=2-A1(n,1)+round(A1(n,1)/4);
    t_jd(n,1)=round(365.25*(year(n,1)+4716))+round(30.6001*(month(n,1)+1))+day(n,1)+B1(n,1)-1524.5;
    t_jdc(n,1)=((t_jd(n,1)+t_gmt(n,1))-2451545)/36525;
%  b. Solar position relative to Earth    
    S(n,1)=21.448-t_jdc(n,1)*(46.815+t_jdc(n,1)*(0.00059-(t_jdc(n,1)*0.001813)));
    O_ob_mean(n,1)=23+((26+(S(n,1)/60))/60);
    O_ob(n,1)=O_ob_mean(n,1)+0.00256*cos(125.04-1934.136*t_jdc(n,1)*pi/180);
    E_c(n,1)=0.016708634-t_jdc(n,1)*(0.000042037+0.0000001267*t_jdc(n,1));
    O_ls_mean(n,1)=mod(280.46646+t_jdc(n,1)*(36000.76983+0.0003032*t_jdc(n,1)),360);
    O_as_mean(n,1)=357.52911+t_jdc(n,1)*(35999.05029-0.0001537*t_jdc(n,1));
    a(n,1)=(O_as_mean(n,1)*pi/180);
    b(n,1)=sin(a(n,1));
    c(n,1)=sin(b(n,1)*2);
    d(n,1)=sin(c(n,1)*3);
    O_cs(n,1)=b(n,1)*(1.914602-t_jdc(n,1)*(0.004817+0.000014*t_jdc(n,1)))+c(n,1)*(0.019993-0.000101*t_jdc(n,1))+d(n,1)*0.000289;
    O_ls(n,1)=O_ls_mean(n,1)+O_cs(n,1);
    O_al(n,1)=O_ls(n,1)-0.00569-(0.00478*sin((125.04-1934.136*t_jdc(n,1))*pi/180));
    solar_dec(n,1)=asin(sin(O_ob(n,1)*pi/180)*sin(O_al(n,1)*pi/180))*180/pi;
    O_ta(n,1)=O_as_mean(n,1)+O_cs(n,1);
    A(n,1)=(tan(O_ob(n,1)*pi/360))^2;
    B(n,1)=sin(2*O_ls_mean(n,1)*pi/180);
    C(n,1)=sin(O_as_mean(n,1)*pi/180);
    D(n,1)=cos(2*O_ls_mean(n,1)*pi/180);
    E(n,1)=sin(4*O_ls_mean(n,1)*pi/180);
    F(n,1)=sin(2*O_as_mean(n,1)*pi/180);
    E_t(n,1)=4*(A(n,1)*B(n,1)-2*E_c(n,1)*C(n,1)+4*E_c(n,1)*A(n,1)*C(n,1)*D(n,1)-0.5*(A(n,1)^2)*E(n,1)-(4/3)*E_c(n,1)^2*F(n,1))*(180/pi);
    lstm=15*round(lon/15);
    t_s(n,1)=((hour(n,1)*60)+minute(n,1))+4*(lstm-lon)+E_t(n,1); %(60*hour(n,1))+minute(n,1)+(s/60)+E_t(n,1)-4*lon+60*t_zone;
    O_ha(n,1)=t_s(n,1)/4-180;
%  c. Solar position relative to stream position    
    m(n,1)=sin(lat*(pi/180))*sin(solar_dec(n,1)*(pi/180))+cos(lat*(pi/180))*cos(solar_dec(n,1)*(pi/180))*cos(O_ha(n,1)*(pi/180));
    sol_zen(n,1)=acos(m(n,1))*(180/pi);
end

% Fresnel's Reflectivity
n=1.333;
alpha_rad=sol_zen*(pi/180);
for i=1:length(alpha_rad)
    if ((alpha_rad(i,1))<pi/2)
        beta_rad(i,1)=asin(sin(alpha_rad(i,1))/n);
        a(i,1)=(tan(alpha_rad(i,1)-beta_rad(i,1)))^2;
        b(i,1)=(tan(alpha_rad(i,1)+beta_rad(i,1)))^2;
        c(i,1)=(sin(alpha_rad(i,1)-beta_rad(i,1)))^2;
        d(i,1)=(sin(alpha_rad(i,1)+beta_rad(i,1)))^2;
        a_deg=a*(180/pi);
        b_deg=b*(180/pi);
        c_deg=c*(180/pi);
        d_deg=d*(180/pi);
        ah(i,1)=0.5*((a_deg(i,1)/b_deg(i,1))+(c_deg(i,1)/d_deg(i,1)));
    else
        ah(i,1)=1;
    end
end

sol_refl(:,1)=interp1(time_met(:,1),ah(:,1),time_mod(:,1),'pchip');
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