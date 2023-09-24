'''
Author:
Michael Scoleri
Violet Shi 
James Gallagher 

Data: 09/03/2023

Filename: interpolation.py

Function: 
Provides an equivalent function in Python for Matlab's interp1 function.
'''
import scipy
import numpy as np

'''
The interpolate function in matlab, which has the following structure:
interp1(x, v, xq, method), where... 
    x is the x-values of the known points
    v is the y-values of the known points
    xq is all the points we want to know
    method specifies how to fit the data (linear, cubic, nearest, etc.)

In order to translate this into Python, we use scipy's interp1d method.
This method takes the following parameters
interp1d(x, y, kind), where...
    x is the x-values
    y is the y-values
    kind specifies how to fit the data (linear, cubic, nearest, etc.)

This interp1 function returns a function. We have to pass the data we want 
to fit to this function. We can accomplish the matlab function 
with a two step python function:

    f = interp1d(x, y, kind)                    # This gives us an interpolation function
    array = []
    for query_point in xq:
        array.append(f(query_point))            # This gives us a 1D array

We can use scipy's interp1d function to accomplish the same thing as Matlab's interp1 function
'''

'''
An equivalent interpolation function to matlab
Returns a 1-D array of interpolated values
'''

### Default method is linear
def interpolation(x, y, xq, method = "linear"):
    return scipy.interpolate.interp1d(x, y, method)(xq)

def pchipinterpolation(x, y, xq):
    return scipy.interpolate.PchipInterpolator(x, y)(xq)
