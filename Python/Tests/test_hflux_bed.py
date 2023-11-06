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


def test_hflux_bed():
    depth_of_measure = np.array([2] * TEST_ARRAY_LENGTH)
    water_temp = np.array([17.443] * TEST_ARRAY_LENGTH)
    bed_temp = np.array([12] * TEST_ARRAY_LENGTH)

    width_m = np.array(
        [
            4.89149604764764,
            4.86849618992830,
            4.84548770754097,
            4.82247060434835,
            4.79944488420915,
            4.77641055097814,
            4.75336760850614,
            4.73031606064005,
            4.70725591122288,
            4.68418716409361,
            4.66110982308743,
            4.63802389203550,
            4.61492937476515,
            4.59182627509980,
            4.56871459685897,
            4.54559434385827,
            4.52246551990949,
            4.49932812882045,
            4.47618217439520,
            4.45302766043385,
            4.42986459073270,
            4.40669296908421,
            4.38351279927692,
            4.36032408509559,
            4.33712683032115,
            4.31392103873068,
        ]
    )

    wp_m = np.array(
        [
            4.89454170600277,
            4.87155730185408,
            4.84856442494694,
            4.82556308139676,
            4.80255327735974,
            4.77953501903399,
            4.75650831266069,
            4.73347316452529,
            4.71042958095876,
            4.68737756833873,
            4.66431713309099,
            4.64124828169065,
            4.61817102066363,
            4.59508535658808,
            4.57199129609579,
            4.54888884587375,
            4.52577801266575,
            4.50265880327390,
            4.47953122456040,
            4.45639528344917,
            4.43325098692771,
            4.41009834204888,
            4.38693735593276,
            4.36376803576867,
            4.34059038881720,
            4.31740442241223,
        ]
    )

    sed_type = np.array(
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    )

    hf = HeatFluxCalculations()
    bed = hf._hflux_bed(sed_type, water_temp, bed_temp, depth_of_measure, width_m, wp_m)

    ### round to 10 digits for consistency across programs
    correct = True

    matlab_output = [
        -3.81247233410512,
        -3.81249563555017,
        -3.81251928195798,
        -3.81254328017086,
        -3.81256763720155,
        -3.81259236023842,
        -3.81261745665070,
        -3.81264293399404,
        -3.81266880001613,
        -3.81269506266264,
        -3.81272173008327,
        -3.81274881063813,
        -3.81277631290424,
        -3.81280424568234,
        -3.81283261800394,
        -3.81286143913857,
        -3.81289071860141,
        -3.81292046616111,
        -3.81295069184794,
        -3.81298140596221,
        -3.81301261908312,
        -3.81304434207777,
        -3.81307658611070,
        -3.81310936265365,
        -3.81314268349580,
        -3.81317656075434,
    ]

    correct = True

    for i in range(len(bed)):
        ours = bed[i]
        matlab = matlab_output[i]
        if abs(ours - matlab) > ACCEPTABLE_ERROR:
            print(ours, matlab)
            correct = False

    assert correct


test_hflux_bed()
