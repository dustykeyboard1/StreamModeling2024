import numpy as np
import sys
import os
import pytest

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.shortwave_reflection_calculations import (
    ShortwaveReflectionCalculations,
)

TEST_ARRAY_LENGTH = 10
ACCEPTABLE_ERROR = 1e-10


def test_shortwave_refl():
    year = np.array([2012] * 10)
    month = np.array([6] * 10)
    day = np.array([13] * 10)
    minute = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    hour = np.array([17] * 10)
    time_mod = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_met = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    t_zone = 5
    lon = -76.067000000000000
    lat = 43.030000000000000

    assert (
        (len(year) == TEST_ARRAY_LENGTH)
        and (len(month) == TEST_ARRAY_LENGTH)
        and (len(day) == TEST_ARRAY_LENGTH)
        and (len(minute) == TEST_ARRAY_LENGTH)
        and (len(time_mod) == TEST_ARRAY_LENGTH)
        and (len(time_met) == TEST_ARRAY_LENGTH)
    )

    sr = ShortwaveReflectionCalculations()
    ah = sr.hflux_shortwave_refl(
        year, month, day, hour, minute, lat, lon, t_zone, time_met, time_mod
    )

    matlab_output = np.array(
        [
            0.0830696531751561,
            0.0842738509457202,
            0.0855026910279460,
            0.0867555043928942,
            0.0880316220116252,
            0.0893303748551999,
            0.0906529002407182,
            0.0920007592357054,
            0.0933739184368597,
            0.0947723444408794,
        ]
    )

    correct = True

    for i in range(len(ah)):
        ours = ah[i]
        matlab = matlab_output[i]
        if abs(ours - matlab) > ACCEPTABLE_ERROR:
            print(ours, matlab)
            correct = False

    assert correct


test_shortwave_refl()
