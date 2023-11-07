"""
Author: Michael Scoleri
Date: 09-18-23
File: test_hflux_errors.py
Functionality: Test the performance of hflux_errors.py
"""

import sys
import os
import pytest
import numpy as np

# Find the root directory dynmimically. https://stackoverflow.com/questions/73230007/how-can-i-set-a-root-directory-dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)
from Python.src.Core.hflux_errors import handle_errors


def test_valid_arguments():
    """Test if `handle_errors` accepts 6 or 7 arguments without raising exceptions."""
    time_mod = np.array([[1], [2], [3]])
    dist_mod = np.array([[1], [2], [3]])
    temp_mod = np.array([[1, 2], [3, 4], [5, 6]])
    dist_temp = np.array([[1], [2], [3]])
    time_temp = np.array([[1], [2], [3]])
    temp = np.array([[1, 2], [3, 4], [5, 6]])

    handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp)

    handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, True)


def test_invalid_number_of_arguments():
    """Test if handle_errors will raise a ValueError exception when given incorrect number of arguments"""
    with pytest.raises(ValueError):
        handle_errors(1, 2, 3)


def test_invalid_column_vector():
    """Test if handle_errors will raise a TypeError exception when given the incorrect shape of Array"""
    time_mod = np.array([1, 2, 3])
    dist_mod = np.array([[1], [2], [3]])
    temp_mod = np.array([[1, 2], [3, 4], [5, 6]])
    dist_temp = np.array([[1], [2], [3]])
    time_temp = np.array([[1], [2], [3]])
    temp = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(TypeError):
        handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp)
