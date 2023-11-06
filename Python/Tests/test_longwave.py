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


def test_longwave():
    air_temp = np.array([20] * TEST_ARRAY_LENGTH)
    rel_hum = np.array([55] * TEST_ARRAY_LENGTH)
    water_temp = np.array([17.443] * TEST_ARRAY_LENGTH)
    cl = np.array([0.3125] * TEST_ARRAY_LENGTH)

    # vts is the only array that changes
    vts = np.empty(0)
    for x in range(750, 801, 2):
        vts = np.append(vts, [float(x / 1000.0)])

    assert len(vts) == TEST_ARRAY_LENGTH

    hf = HeatFluxCalculations()
    longwave, atm_rad, back_rad, land_rad = hf._hflux_longwave(
        air_temp, water_temp, rel_hum, cl, vts
    )

    matlab_output = [
        -47.7850425827384,
        -47.9064947905216,
        -48.0279469983048,
        -48.1493992060879,
        -48.2708514138711,
        -48.3923036216543,
        -48.5137558294374,
        -48.6352080372206,
        -48.7566602450037,
        -48.8781124527869,
        -48.9995646605700,
        -49.1210168683532,
        -49.2424690761364,
        -49.3639212839195,
        -49.4853734917027,
        -49.6068256994858,
        -49.7282779072690,
        -49.8497301150522,
        -49.9711823228353,
        -50.0926345306185,
        -50.2140867384017,
        -50.3355389461848,
        -50.4569911539680,
        -50.5784433617511,
        -50.6998955695343,
        -50.8213477773174,
    ]

    correct = True

    for i in range(len(longwave)):
        ours = longwave[i]
        matlab = matlab_output[i]
        if abs(ours - matlab) > ACCEPTABLE_ERROR:
            print(ours, matlab)
            correct = False

    assert correct


test_longwave()
