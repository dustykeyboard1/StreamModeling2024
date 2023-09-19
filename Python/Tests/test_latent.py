import numpy as np
import sys
import os
import pytest

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.hflux_latent import hflux_latent
from Python.src.Heat_Flux.hflux_longwave import hflux_longwave

def test_latent():

    ### THIS TEST TAKES SHORTWAVE AND MATLAB INPUT FROM THE FIRST TIME THAT HFLUX_LATENT IS CALLED FROM HFLUX_FLUX
    ### OTHERWISE, WE GET INCONSISTENT DATA (inconsistent meaning the shortwave values do not align with other inputs, not
    ### other data somehow breaks this) (TL:DR; it works)

    air_temp = np.array([20] * 26)
    rel_hum = np.array([55] * 26)
    water_temp = np.array([17.443] * 26)
    wind_speed = np.array([0] * 26)
    cl = np.array([.3125] * 26)
    z = 150
    eq2 = 1
    ### Constants above

    # vts is the only array that changes
    vts = np.empty(0)
    for x in range(750, 801, 2):
        vts = np.append(vts, [float(x / 1000.0)])
    
    ### here we use our hflux_longwave to get good longwave data. Pretty cool!
    
    longwave, _, _, _ = hflux_longwave(air_temp, rel_hum, water_temp, vts, cl)
    shortwave = [36.4479812862875,
36.5451759030510,
36.6423705198144,
36.7395651365778,
36.8367597533413,
36.9339543701047,
37.0311489868682,
37.1283436036316,
37.2255382203950,
37.3227328371585,
37.4199274539219,
37.5171220706853,
37.6143166874488,
37.7115113042122,
37.8087059209756,
37.9059005377391,
38.0030951545025,
38.1002897712659,
38.1974843880294,
38.2946790047928,
38.3918736215562,
38.4890682383196,
38.5862628550831,
38.6834574718465,
38.7806520886100,
38.8778467053734]
    
    # print(shortwave, longwave)
        
    latent = hflux_latent(shortwave, longwave, rel_hum, water_temp, wind_speed, air_temp, z, eq2)
    # print(latent)
    
    matlab_output = [6.99729041271735,
7.01399363313129,
7.03069685354526,
7.04740007395921,
7.06410329437316,
7.08080651478713,
7.09750973520107,
7.11421295561502,
7.13091617602898,
7.14761939644292,
7.16432261685686,
7.18102583727082,
7.19772905768480,
7.21443227809873,
7.23113549851269,
7.24783871892663,
7.26454193934057,
7.28124515975456,
7.29794838016849,
7.31465160058248,
7.33135482099641,
7.34805804141035,
7.36476126182428,
7.38146448223822,
7.39816770265221,
7.41487092306614]
    
    ### round to 10 digits for consistency across programs
    correct = True

    for i in range(len(latent)):
        ours = str(round(latent[i], 10))
        matlab = str(round(matlab_output[i], 10))
        if ours != matlab:
            print(ours, matlab)
            correct = False
    
    assert correct
    print(correct)
    
test_latent()
    