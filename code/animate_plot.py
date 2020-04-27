#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:44:11 2020

@author: mohit
"""

from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


fig = plt.figure()
ax = p3.Axes3D(fig)

q = [[0], [0], [0]]


x = np.array(model.c)
y = np.array(model.w)
z = np.array(model.l)


ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(-1, 1)

points, = ax.plot(x, y, z, "*")
txt = fig.suptitle("")


def update_points(num, x, y, z, points):
    txt.set_text("num={:d}".format(num))  # for debug purposes

    new_x = x + np.random.normal(1, 0.1, size=(len(x),))
    new_y = y + np.random.normal(1, 0.1, size=(len(y),))
    new_z = z + np.random.normal(1, 0.1, size=(len(z),))
    print(new_x, new_y)
    points.set_data(new_x, new_y)
    points.set_3d_properties(new_z, "z")

    return points, txt


ani = animation.FuncAnimation(fig, update_points, frames=10, fargs=(x, y, z, points))

plt.show()
