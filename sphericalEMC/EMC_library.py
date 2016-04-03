#   file: EMC_library.py
#
#   This code contains several functions used in the spherical EMC algorithm.
#   See also: sphericalEMC.py
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
from math import pi
from decimal import *


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


def intensity_mapping(qi, theta, phi, listRj):
    ''' constructs tomographic grid from detector coordinates and rotation
    sampling.
    '''

    M = len(listRj)
    N = len(qi)
    qimap = np.zeros(N*M)
    thetamap = np.zeros(N*M)
    phimap = np.zeros(N*M)

    xi, yi, zi = sph2cart(qi, theta, phi)
    for j in range(M):
        Rj = listRj[j]
        ximapj = Rj[0, 0]*xi + Rj[0, 1]*yi + Rj[0, 2]*zi
        yimapj = Rj[1, 0]*xi + Rj[1, 1]*yi + Rj[1, 2]*zi
        zimapj = Rj[2, 0]*xi + Rj[2, 1]*yi + Rj[2, 2]*zi

        qimapj, thetamapj, phimapj = cart2sph(ximapj, yimapj, zimapj)

        qimap[j*N:(j+1)*N] = qimapj
        thetamap[j*N:(j+1)*N] = thetamapj
        phimap[j*N:(j+1)*N] = phimapj

    return qimap, thetamap, phimap
###############################################################################
###############################################################################


def sampling_weights(L):

    ''' In pseudo-TeX, the formula for the bandwidth L weights is
     w_L(j) = 2/L sin((pi*(2j+1))/(4L)) *
    sum_{k=0}^{L-1} 1/(2k+1)*sin((2j+1)(2k+1)pi/(4L))'''

    id_val = np.arange(0, 2*L, 1)

    w = np.zeros(2*L)
    for k in range(L):
        w += 1./(2*k+1)*np.sin((2*id_val+1)*(2*k+1)*pi/(4*L))
    w = w*2./L*np.sin(pi*(2*id_val+1)/(4*L))

    # needed because sum of w equal to 2
    w = w/2.0
    return w


def sampling_weights_total(L):

    w = np.zeros(8*L**3)
    wk = sampling_weights(L)
    i = 0
    for j1 in range(2*L):
        for k in range(2*L):
            for j2 in range(2*L):
                w[i] = (1./(2*L)**2)*wk[k]
                i += 1
    return w


def samplingGridFFT(L):
    '''sampling grid to perform FFT on SO3 for a L-bandlimited function.
    It is given in zyz-Euler angles, (alpha,beta,gamma)
    '''

    id_val = np.arange(0, 2*L, 1)
    alpha = 2*pi/(2*L)*id_val
    beta = pi/(4*L)*(2*id_val+1)
    gamma = 2*pi/(2*L)*id_val

    euler = np.zeros((8*L**3, 3))
    i = 0
    for j1 in range(2*L):
        for k in range(2*L):
            for j2 in range(2*L):
                euler[i, 0] = alpha[j1]
                euler[i, 1] = beta[k]
                euler[i, 2] = gamma[j2]
                i += 1
    return euler


def euler2matrix(theta1, theta2, theta3):

    ''' return matrix from zyz-Euler angles -- see eg. chap.3 in MPhil Thesis,
    available at https://minerva-access.unimelb.edu.au/handle/11343/58637
    '''

    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    c3 = np.cos(theta3)

    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    s3 = np.sin(theta3)
    R = np.zeros((3, 3))

    R[0, 0] = c1*c2*c3-s1*s3
    R[1, 0] = s1*c2*c3+c1*s3
    R[2, 0] = -s2*c3

    R[0, 1] = -c1*c2*s3-s1*c3
    R[1, 1] = -s1*c2*s3+c1*c3
    R[2, 1] = s2*s3

    R[0, 2] = c1*s2
    R[1, 2] = s1*s2
    R[2, 2] = c2

    return R


def getSamplingRotation(L):
    '''
    Returns rotations matrices for sampling on the rotation group needed in
    the EMC algorithm.
    For a bandlimit L, we have 8L**3 samples on SO(3)
    '''

    eulerAngles = samplingGridFFT(L)
    M = np.size(eulerAngles, 0)
    listRj = []
    for j in range(M):
        listRj.append(euler2matrix(eulerAngles[j, 0], eulerAngles[j, 1], eulerAngles[j, 2]))

    wj = sampling_weights_total(L)
    return listRj, wj


###############################################################################
###############################################################################
def safe_ln(x, minval=0.00000000000001):
    return np.log(x.clip(min=minval))


def compute_Pjk_matrix(MPiik, MIj, inds_k_non_zero, inds_k_zero, wj):

    M = np.size(MIj, 1)
    K = np.size(MPiik, 1)

    MPjk_dec = np.zeros((M, K), dtype=object)
    MPjk = np.zeros((M, K))

    # poisson !!!!!! PROBLEM TOO HIGH VALUES OR TWO LOW -> solved by \
    # Decimal and log computation of the probability
    for k in range(K):
        S = 0
        for j in range(M):
            Pjk = 0

            Pjk += (MPiik[inds_k_non_zero[k], k] * safe_ln(MIj[inds_k_non_zero[k],j])).sum()
            Pjk += -MIj[:, j].sum()

            MPjk_dec[j, k] = Decimal(Pjk).exp()
            MPjk_dec[j, k] = Decimal(wj[j])*MPjk_dec[j, k]

            S += MPjk_dec[j, k]
        MPjk[:, k] = (MPjk_dec[:, k]/S).astype(float)

    return MPjk
