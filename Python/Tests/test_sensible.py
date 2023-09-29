import numpy as np
import sys
import os
import pytest

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.hflux_sensible import hflux_sensible

def test_sensible():

    z = 150
    air_temp = np.array([20] * 26)
    rel_hum = np.array([55] * 26)
    water_temp = np.array([17.443] * 26)
    wind_speed = np.array([0] * 26)
    ### The above never changes in the sample ###

    ## since eq3 = 2 depends on wind speed, and wind speed is always 0, its trivially 0
    ## will ask her for more data
    eq3 = 1
    latent = np.array([6.99729041271735,
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
7.41487092306614])
    
    matlab_output = [-1.20762021014190,
-1.21050291835559,
-1.21338562656929,
-1.21626833478298,
-1.21915104299668,
-1.22203375121037,
-1.22491645942406,
-1.22779916763776,
-1.23068187585145,
-1.23356458406515,
-1.23644729227884,
-1.23933000049253,
-1.24221270870623,
-1.24509541691992,
-1.24797812513362,
-1.25086083334731,
-1.25374354156100,
-1.25662624977470,
-1.25950895798839,
-1.26239166620209,
-1.26527437441578,
-1.26815708262948,
-1.27103979084317,
-1.27392249905686,
-1.27680520727056,
-1.27968791548425]

    sensible = hflux_sensible(water_temp, air_temp, rel_hum, wind_speed, z, latent, eq3)

    ### round to 10 digits for consistency across programs

    for i in range(len(sensible)):
        ours = str(round(sensible[i], 10))
        matlab = str(round(matlab_output[i], 10))
        if ours != matlab:
            print(ours, matlab)
            assert False
            
    assert True


test_sensible()