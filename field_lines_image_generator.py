from __future__ import division
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt,sin,cos,atan2, sqrt, pi
from scipy.special import ellipk, ellipe
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# Variables
duration_steps = 5e3#1e6
b_comp = np.array([0.0,0.0,.07712710231773079])


# Constants/Refrences
generic_xyz_coords = np.array([0,0,0])
mu_0 = 1.25663706e-6


class coil:
    def __init__(self, radius = 0.05, current=100, position=np.array([0.0, 0.0, 0.0]), plane_axis = 0, num_turns = 1, usedecimal = False):
        self.radius = float(radius)
        self.current = float(current)
        self.position = np.float64(position)
        self.num_turns = int(num_turns)
        self.B_0 = self.current * float(mu_0) / (2*self.radius)
        self.current_2 = self.current*self.current
        self.ellipk = ellipk
        self.ellipe = ellipe
        self.atan2 = atan2
        self.sqrt = sqrt
        self.pi = pi


    def get_components_from_pos(self, x,y,z, usedecimal = False):
        z = z - self.position[2] # Shift only works for vertically stacked coils
        r = self.sqrt(x*x + y*y)
        a = r/self.radius
        B = z/self.radius
        gamma = z/r
        Q = ((1.0+a) * (1.0+a) + B*B)
        #            k = self.sqrt(4*a/Q)
        k = (4*a/Q)



        E_k = self.ellipe(k)
        K_k = self.ellipk(k)

        B_z = self.B_0 * (1.0)/(self.pi*self.sqrt(Q)) * ( (E_k * (1.0 - a*a - B*B)/(Q-4.0*a)) + K_k )
        B_r = self.B_0 * (gamma)/(self.pi*self.sqrt(Q)) * (E_k * (1.0 + a*a + B*B)/(Q-4.0*a) - K_k )

        B_x = x/r*B_r
        B_y = y/r*B_r

        #return np.array([B_x, B_y, B_z])
        return B_x, B_z


first_coil = coil(radius = 0.035,
                  current =100.0, # Currently num turns does nothing!!
                  position=np.array([0.0,0.0,0.0]),
                  plane_axis = 0,
                  num_turns = 1,
                  usedecimal = False
                  )

second_coil = coil(radius = 0.035,
                  current = -100.0, # Currently num turns does nothing!!
                  position=np.array([0.0,0.0,0.05]),
                  plane_axis = 0,
                  num_turns = 1,
                  usedecimal = False
                  )
gcfp1 = np.vectorize(first_coil.get_components_from_pos)
gcfp2 = np.vectorize(second_coil.get_components_from_pos)

def gcfp(x,y,z):
    a,b = first_coil.get_components_from_pos(x,y,z)
    c,d = second_coil.get_components_from_pos(x,y,z)
    return (a+c), (b+d)

gcfp = np.vectorize(gcfp)

x = np.linspace(-.05, .05, 200)
y = np.linspace(-.025, .075, 200)
xx, yy = np.meshgrid(x, y)
B_x_1, B_y_1 = gcfp1(xx,0,yy)
B_x_2, B_y_2 = gcfp2(xx,0,yy)

B_x = B_x_1+B_x_2
B_y = B_y_1+B_y_2
#z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

plt.figure()
#plt.contourf(x,y,z)
plt.margins(0, 0)

color = np.log(B_x**2 + B_y**2)

strm = plt.streamplot(x, y, B_x, B_y, density = 3, cmap='autumn')
#plt.colorbar(strm.lines)
#plt.clim(vmax = .75, vmin = -.75)
plt.figure()
plt.contour(x,y,color)
plt.show()
