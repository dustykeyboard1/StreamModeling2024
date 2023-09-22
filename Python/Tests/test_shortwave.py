import numpy as np
import sys
import os
import pytest

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.hflux_shortwave import hflux_shortwave

def test_shortwave():

    eq1 = 1
    sol_refl = .5
    solar_rad = np.array([17.8] * 26)
    ### The above never changes in the sample ###

    sol_in = np.empty(0)
    for i in range(133500, 142401, 356):
        sol_in = np.append(sol_in, [i / 10000.0])
    
    shade = np.empty(0)
    for i in range(2500, 1999, -20):
        shade = np.append(shade, [i / 10000.0])

    shortwave = hflux_shortwave(solar_rad, shade, sol_refl, eq1)

    print(shortwave)


test_shortwave()