import numpy as np
import sys
import os
import pytest

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Heat_Flux.heatflux_calculations import HeatFluxCalculations

TEST_ARRAY_LENGTH = 26
ACCEPTABLE_ERROR = 1e-10


def test_shortwave():
    eq1 = 1
    sol_refl = 0.0831
    solar_rad = np.array([53] * TEST_ARRAY_LENGTH)

    shade = np.array(
        [
            0.250000000000000,
            0.248000000000000,
            0.246000000000000,
            0.244000000000000,
            0.242000000000000,
            0.240000000000000,
            0.238000000000000,
            0.236000000000000,
            0.234000000000000,
            0.232000000000000,
            0.230000000000000,
            0.228000000000000,
            0.226000000000000,
            0.224000000000000,
            0.222000000000000,
            0.220000000000000,
            0.218000000000000,
            0.216000000000000,
            0.214000000000000,
            0.212000000000000,
            0.210000000000000,
            0.208000000000000,
            0.206000000000000,
            0.204000000000000,
            0.202000000000000,
            0.200000000000000,
        ]
    )

    assert len(shade) == TEST_ARRAY_LENGTH

    hf = HeatFluxCalculations()
    shortwave = hf._hflux_shortwave(solar_rad, shade, sol_refl, eq1)

    matlab_output = [
        36.4479812862875,
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
        38.8778467053734,
    ]

    assert len(matlab_output) == TEST_ARRAY_LENGTH

    correct = True

    for i in range(len(shortwave)):
        ours = shortwave[i]
        matlab = matlab_output[i]
        if abs(ours - matlab) > ACCEPTABLE_ERROR:
            print(ours, matlab)
            correct = False


test_shortwave()
