'''
Author:
Michael Scoleri
Violet Shi 
James Gallagher 

Data: 09/04/2023

Filename: graphing_example.py

Function: 
Demonstrates the 3-dimensional graphing capabilities.
'''

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

def create3dshape(x, y):
    '''
    Create 3-dimensional Sine function
    Params: x - np array generated from np.linespace
            y - np array generated from np.linespace
    Returns: np array generated from x and y
    '''
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = create3dshape(X, Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Demo 3D Plot')

plt.show()