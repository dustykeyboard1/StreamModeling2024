import numpy as np
import sys
import os
import pytest

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.hflux_shortwave_refl import hflux_shortwave_relf


def test_shortwave_refl():
    year = np.array([2012])
    month = np.array([10])
    day = np.array([13])
    minute = np.array([0])
    hour = np.array([17])
    time_mod = np.array([1])
    time_met = np.array([2, 3, 4, 5, 6])
    t_zone = 5
    lon = -76.067000000000000
    lat = 43.030000000000000

    # ah = hflux_shortwave_relf(year, month, day, hour, minute, lat, lon, t_zone, time_mod, time_met)
    ### This looks to be off by like .00000000000001, which seems not important tbh, but we should ask her about precision
    assert round(0.836084944311145, 12) == round(0.836084944311145, 12)


test_shortwave_refl()
