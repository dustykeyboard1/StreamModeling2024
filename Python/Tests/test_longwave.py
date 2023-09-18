import numpy as np
import sys
import os
import pytest

#Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.hflux_longwave import hflux_longwave

def test_longwave():

    air_temp = np.array([20, 20, 20, 20])
    rel_hum = np.array([55, 55, 55, 55])
    water_temp = np.array([17.443, 17.443, 17.443, 17.443])
    vts = np.array([.75, .75, .75, .75])
    cl = np.array([.3125, .3125, .3125, .3125])
    
    longwave, atm_rad, back_rad, land_rad = hflux_longwave(air_temp, rel_hum, water_temp, vts, cl)
    print(longwave, atm_rad, back_rad, land_rad)

test_longwave()
    
    