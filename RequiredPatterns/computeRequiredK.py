#   file: computeRequiredK.py
#
#   This code computes an estimate of the minimum number of diffraction pattern
#   required to reconstruct the intensity on a shell of given radius and MPC.
#   See for instance Fig. 3  and section V-A-1 in our ArXiv preprint
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

import numpy as np
import healpy as hp
from math import pi


def L2nside(L):

    ''' Gives the closest nside corresponding to L, such that L <=2nside+1'''

    a = np.log((L-1.0)/2)/np.log(2)

    nside = int(2**(np.ceil(a)))

    return nside

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

#############################################################################


def requiredPatternsMPC(shellIndex, nside, MPC):
    ''' Returns an estimate of the number of required patterns K to visit all
    pixels on the shell of index shellIndex. The parameter nside controls the
    grid resolution, and therefore is linked directly to the value of L
    available. The Mean Photon Count (MPC) parameter controls the level of
    signal available.
    '''
    # define detector characteristics and experimental conditions
    wavelength = 2  # in Angstroms
    zD = 1e-2  # detector distance, in meters
    pw = 3.38e-5  # pixel size, in meters

    qmax = 4*pi/wavelength
    Deltaq = qmax*np.sin(0.5*np.arctan(pw*1/zD))  # pixel spacing at the origin

    q_shell = qmax*np.sin(0.5*np.arctan(pw*shellIndex/zD))
    Npix_dec = int(np.ceil(2*pi*q_shell/Deltaq))

    # coords detector in reciprocal space
    qdec = np.ones(Npix_dec)*q_shell
    thetadec = pi/2-np.arcsin(qdec/qmax)
    phidec = np.linspace(0, 2*pi, Npix_dec+1)[:Npix_dec]
    xdec, ydec, zdec = sph2cart(qdec, thetadec, phidec)

    npix = hp.nside2npix(nside)
    # generate nbSamples uniform rotations in SO(3)
    histo = np.zeros(npix)
    k = 0
    while (len(np.nonzero(histo)[0]) != npix):
        quat = randS3(1)
        xr, yr, zr = rotate(xdec, ydec, zdec, quat)

        q, theta, phi = cart2sph(xr, yr, zr)

        randomCounts = np.random.poisson(np.ones(Npix_dec)*MPC)
        validPixels = np.where(randomCounts > 0)

        inds = hp.ang2pix(nside, theta[validPixels], phi[validPixels])

        histo[inds] += 1
        k += 1

    return float(k)

#############################################################################
nsideList = np.array([1, 2, 4, 8, 16, 32, 64])
MC = 50  # number of Monte Carlo realizations
shells = np.array([40])
requiredK = np.zeros((np.size(nsideList), np.size(shells)))

# define detector characteristics and experimental conditions
wavelength = 2  # in Angstroms
zD = 1e-2  # in meters
pw = 3.38e-5  # in meters

qmax = 4*pi/wavelength
Deltaq = qmax*np.sin(0.5*np.arctan(pw*1/zD))  # pixel spacing at the origin

MPC = 10  # Mean Photon Count on the shell
for ids, s in enumerate(shells):
    # compute Lmax
    q_shell = qmax*np.sin(0.5*np.arctan(pw*s/zD))
    Lmax = np.floor(1+pi*q_shell/Deltaq)
    nsideMax = L2nside(Lmax)
    print(Lmax)
    for idnside, nside in enumerate(nsideList):
        if nside < nsideMax + 1:
            k = 0
            print(nside)
            for n in range(MC):
                k += requiredPatternsMPC(s, nside, MPC)
            requiredK[idnside, ids] = k/MC

    #np.save('requiredKMPC'+str(MPC)+'.npy', requiredK)
