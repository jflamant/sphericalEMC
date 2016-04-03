#   file: energyShellDistribution.py
#
#   This code computes the energy among spherical harmonic coefficients for
#   successive shells. See for instance Fig. 2 (bottom) in our ArXiv preprint
#                       http://arxiv.org/abs/1602.01301
#
#   Copyright (c) J. Flamant, April 2016.
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, please visit http://www.gnu.org


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from math import pi


wavelength = 2  # in Angstroms
D = 1e-2  # detector distance, in meters
px = 3.38e-5  # pixel size, in meters
re = 2.8179403267*1e-15  # in m
nphotons = 1e13
beamarea = 1e-14  # in m**2

qmax = 4*pi/wavelength  # largest scattering vector
grid_size = 64  # number of pixels in the upper half detector

nside = 128
lmax = 3*nside-1  # if running anafast in iterative mode, otherwise use 2*nside

l_even = np.arange(0, lmax, 2)  # only even l, due to Friedel symmetry

shell_index = np.arange(1, grid_size+1, 1)

E = np.zeros((lmax+1, grid_size))
Llim = np.zeros(grid_size)
thresh = 1e-5  # a threshold to obtain the bandlimit
for s in range(1, grid_size+1, 1):
    print(s)
    I = np.load('numpy_array_of_intensity_shell_index_s.npy')

    E[:, s-1] = hp.anafast(I, iter=30)
    E[:, s-1] = E[:, s-1]/np.sum(E[:, s-1])
    # find Llim_s (Bandlimit on shell s) (may require tuning, check ouput values)

    x = np.where(E[l_even, s-1] < thresh)
    Llim[s-1] = np.min(x)

plt.imshow(E[l_even, :], norm=col.LogNorm(), cmap='viridis')

plt.colorbar()
plt.show()
