#   file: generatSamplesShell.py
#
#   This code generates a given number of patterns, on a specified shell. The
#   samples are obtained by Poisson realizations of the underlying intensity.
#   The theoretical intensity on the shell (continuous) is to be calculated
#   using 'compute3Dintensityshell.py'.
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

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from math import pi


def sph2cart(qi, theta, phi):

    xi = qi*np.sin(theta)*np.cos(phi)
    yi = qi*np.sin(theta)*np.sin(phi)
    zi = qi*np.cos(theta)

    return xi, yi, zi


def cart2sph(xi, yi, zi):

    qi = np.sqrt(xi**2+yi**2+zi**2)
    theta = np.arccos(zi/qi)
    phi = np.arctan2(yi, xi)

    return qi, theta, phi


def randS3(N=100):

    '''
    Generates N samples from the uniform distribution on S^3
    Input :

     - N: size of the sample

    Ouput:

     - X is a 4 x N array with columns being samples from Unif. dist on S^3

    See Shoemake, K., 1992, July. Uniform random rotations. In Graphics Gems
    III (pp. 124-132). Academic Press Professional, Inc..
    '''

    X = np.zeros((4, N))
    x0 = np.random.rand(1, N)
    theta1 = np.random.rand(1, N)*2*pi
    theta2 = np.random.rand(1, N)*2*pi
    X[0, :] = np.sin(theta1)*np.sqrt(1-x0)
    X[1, :] = np.cos(theta1)*np.sqrt(1-x0)
    X[2, :] = np.sin(theta2)*np.sqrt(x0)
    X[3, :] = np.cos(theta2)*np.sqrt(x0)
    return X


def quaternionMatrix(quat):

    ''' Converts a quaternion to a 3x3 matrix'''

    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    R = np.array([[1-2*q1**2-2*q2**2, 2*(q0*q1 - q2*q3), 2*(q0*q2+q1*q3)],
    [2*(q0*q1 + q2*q3) ,1-2*q0**2-2*q2**2, 2*(q1*q2-q0*q3)],
    [2*(q0*q2 - q1*q3),2*(q0*q3 + q2*q1),1-2*q0**2-2*q1**2]])
    return R


def rotate(x, y, z, q):

    R = quaternionMatrix(q)

    xr = R[0, 0]*x+R[0, 1]*y+R[0, 2]*z
    yr = R[1, 0]*x+R[1, 1]*y+R[1, 2]*z
    zr = R[2, 0]*x+R[2, 1]*y+R[2, 2]*z

    return xr, yr, zr


def generateDataFromShell(shellIndex, nbSamples):

    '''
    This function generates nbSamples Poisson samples from the intensity
    defined on the shellIndex-th shell.

    Default is the 1REI biomolecule. Intensities were calculated numerically
    on a HEALPix grid of parameter nside = 128.
    Parameters:
    - shellIndex: int, shell index of the considered shell
    - nbSamples: int, number of desired samples
    Return: Ndec \times nbSamples real matrix.
    '''

    # load intensity function
    IShell = np.load('path_to_shells_intensities/'+str(shellIndex)+'.npy')
    nside = 128

    # define detector characteristics and experimental conditions
    wavelength = 2  # in Angstroms
    zD = 1e-2  # detector distance, in meters
    pw = 3.38e-5  # pixel size, in meters

    qmax = 4*pi/wavelength
    gridsize = 512  # number of pixels in one direction (total size of the detector 1024*1024)
    qmax_grid = qmax*np.sin(0.5*np.arctan(pw*gridsize/zD)) #largest value of q on the detector 
    Deltaq = qmax*np.sin(0.5*np.arctan(pw*1/zD))  # pixel spacing at the origin

    q_shell = qmax*np.sin(0.5*np.arctan(pw*shellIndex/zD)) # q value at the shellIndex given
    Npix_dec = int(np.ceil(2*pi*q_shell/Deltaq))  # Corresponding number of pixels

    # coords detector in reciprocal space
    qdec = np.ones(Npix_dec)*q_shell
    thetadec = pi/2-np.arcsin(qdec/qmax)
    phidec = np.linspace(0, 2*pi, Npix_dec+1)[:Npix_dec]
    xdec, ydec, zdec = sph2cart(qdec, thetadec, phidec)

    # create output array
    Myk = np.zeros((Npix_dec, nbSamples))

    # generate nbSamples uniform rotations in SO(3)
    quatList = randS3(nbSamples)
    # Get the samples
    for k in range(nbSamples):
        xr, yr, zr = rotate(xdec, ydec, zdec, quatList[:, k])

        q, theta, phi = cart2sph(xr, yr, zr)

        sampledI = hp.get_interp_val(IShell, theta, phi)
        Myk[:, k] = np.random.poisson(sampledI)

    return Myk

np.random.seed(1)  # for reproducibility

# generate 1000 'patterns', from shellIndex=29
Myk = generateDataFromShell(29, 1000) 
plt.plot(Myk)
plt.show()
#np.save('shell7Patterns1000.npy', Myk)
