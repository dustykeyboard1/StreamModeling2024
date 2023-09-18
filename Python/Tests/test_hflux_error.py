import pytest
import numpy as np
from hflux_errors import handle_errors

def test_valid_arguments():
    time_mod = np.array([[1], [2], [3]])
    dist_mod = np.array([[1], [2], [3]])
    temp_mod = np.array([[1, 2], [3, 4], [5, 6]])
    dist_temp = np.array([[1], [2], [3]])
    time_temp = np.array([[1], [2], [3]])
    temp = np.array([[1, 2], [3, 4], [5, 6]])

    # Test with 6 arguments
    handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp)

    # Test with 7 arguments
    handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp, True)

def test_invalid_number_of_arguments():
    with pytest.raises(ValueError):
        handle_errors(1, 2, 3)

def test_invalid_column_vector():
    time_mod = np.array([1, 2, 3])  # Not a column vector
    dist_mod = np.array([[1], [2], [3]])
    temp_mod = np.array([[1, 2], [3, 4], [5, 6]])
    dist_temp = np.array([[1], [2], [3]])
    time_temp = np.array([[1], [2], [3]])
    temp = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(TypeError):
        handle_errors(time_mod, dist_mod, temp_mod, dist_temp, time_temp, temp)
