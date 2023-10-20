"""
Author: Michael Scoleri
Date: 10-19-23
File: test_plotting_errors.py
Functionality: Test the performance of plotting class.
"""

import sys
import os
import pytest
import numpy as np

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Plotting.plotting_class import Plotting as plc
import matplotlib.pyplot as plt


# Initialize the plotting class
p = plc()

# Create some example 2D data arrays
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
z = np.array([[12, 13, 14], [15, 16, 17], [18, 19, 20]])

# Labels and title
title = "3D test plot"
xlab = "XAXIS"
ylab = "YAXIS"
zlab = "ZAXIS"
clabel = "COLORBAR"

fig = p.make3dplot(x, y, z, xlab, ylab, zlab, clabel, title)
fig.show()
input("press enter to close plots.")
