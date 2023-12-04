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
    print(shortwave)

    matlab_output = np.array(
        [
            36.446775,
            36.5439664,
            36.6411578,
            36.7383492,
            36.8355406,
            36.932732,
            37.0299234,
            37.1271148,
            37.2243062,
            37.3214976,
            37.418689,
            37.5158804,
            37.6130718,
            37.7102632,
            37.8074546,
            37.904646,
            38.0018374,
            38.0990288,
            38.1962202,
            38.2934116,
            38.390603,
            38.4877944,
            38.5849858,
            38.6821772,
            38.7793686,
            38.87656,
        ]
    )

    assert len(matlab_output) == TEST_ARRAY_LENGTH

    correct = True

    for i in range(len(shortwave)):
        ours = shortwave[i]
        matlab = matlab_output[i]
        if abs(ours - matlab) > ACCEPTABLE_ERROR:
            print(ours, matlab)
            correct = False

    assert correct


test_shortwave()
