#   file: sphericalEMC.py
#
#   This code performs the (adaptive) spherical EMC algorithm on diffraction
#   patterns generated through 'generateSamplesShell.py'. The roughness of the #   reconstruction can be controlled by the maximum bandlimit Lmax.
#
#   See also: EMC_library.py
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
from EMC_library import *
from math import pi


def LikelihoodImprovement(Likelihood):

    nIt = len(Likelihood)
    relImprovement = np.zeros(nIt-1)
    for i in range(1, nIt):

        relImprovement[i-1] = np.abs(Likelihood[i]-Likelihood[i-1])/np.abs(Likelihood[i-1])

    return relImprovement


def friedel_sym(Ilm, lmax):

    l, m = hp.Alm.getlm(lmax)
    l_odd = np.where(l % 2 == 1)

    Ilm[l_odd] = 0

    return Ilm


def computeLikelihoodPattern(MPjk, MIj, MPiik, wj):

    K = np.size(MPiik, 1)
    M = np.size(MIj, 1)

    Likelihood = np.zeros(K)
    for k in range(K):
        for j in range(M):
            if MPjk[j, k] > 0:
                Likelihood[k] += MPjk[j, k]*(MPiik[:, k]*safe_ln(MIj[:, j]) - MIj[:, j]).sum()
                Likelihood[k] += - MPjk[j, k]*np.log(MPjk[j, k]/wj[j])

    return Likelihood


def L2nside(L):

    ''' Gives the closest nside corresponding to L, such that L <=2nside+1'''

    a = np.log((L-1.0)/2)/np.log(2)

    nside = int(2**(np.ceil(a)))

    return nside


def EMC_shell_coefficients(shellIndex, Ktot, K, Lmax, eta=1e-3, nitmin=4):
    ''' EMC SHELL ALGORITHM with adaptive L choice.
    Input parameters:
    - shellIndex: int, index of the shell considered
    - Ktot: int, total number of diffraction patterns available in the data file
    - K: int, defines the number of patterns to process, ie the K first among Ktot of the data file
    - Lmax: int, maximum bandlimit in the spherical harmonic domain
    - eta: stop criterion for relative likelihood increments (default 1e-3)
    - nitmin: number of iterations of the algorithm for each L value (default 4)
    '''

    ''' DATA ANALYSIS '''
    # load data file, given shellIndex and K patterns
    MPiik = np.load('.data/shell'+str(shellIndex)+'Patterns'+str(Ktot)+'.npy')[:,:K]


    # define monitor quantities
    IlmArray = np.zeros((hp.Alm.getsize(Lmax-1), 100), dtype=complex)
    LikelihoodArray = np.zeros((K, 101))


    # check if there are zero-valued measurements, to increase speed.
    inds_k_non_zero = []
    inds_k_zero = []
    nb_zeropix = 0
    nb_nonzeropix = 0
    for k in range(np.size(MPiik, 1)):
        nonzero = np.where(MPiik[:, k] != 0)[0]
        zero = np.where(MPiik[:, k] == 0)[0]
        inds_k_non_zero.append(nonzero)
        inds_k_zero.append(zero)

        nb_zeropix += len(zero)
        nb_nonzeropix += len(nonzero)
    inds_i_non_zero = []

    for i in range(np.size(MPiik, 0)):
        nonzero = np.where(MPiik[i, :] != 0)[0]
        inds_i_non_zero.append(nonzero)

    print('Nb zero pixels: '+str(nb_zeropix))
    print('Nb non zero pixels: '+str(nb_nonzeropix))
    print('Average photon count per pixel on the shell : '+str(MPiik.mean()))

    ''' ALGORITHM INITIALIZATION'''

    # define detector characteristics and experimental conditions

    wavelength = 2  # in Angstroms
    zD = 1e-2  # detector distance,  in meters
    pw = 3.38e-5  # pixel size, in meters

    qmax = 4*pi/wavelength
    q_shell = qmax*np.sin(0.5*np.arctan(pw*shellIndex/zD))
    Npix_dec = np.size(MPiik, 0)
    # coords detector in reciprocal space
    qdec = np.ones(Npix_dec)*q_shell
    thetadec = pi/2-np.arcsin(qdec/qmax)
    phidec = np.linspace(0, 2*pi, Npix_dec+1)[:Npix_dec]
    xdec, ydec, zdec = sph2cart(qdec, thetadec, phidec)

    # loop on the L values
    i = 0

    likelihood = []
    for L in range(3, Lmax+1, 2):
        nit_resol = 0
        # Creating the tomographic grid

        print('Creating tomographic grid with L = '+str(L))
        listRj, wj = getSamplingRotation(L)
        qimap, thetamap, phimap = intensity_mapping(qdec, thetadec, phidec, listRj)

        ximap, yimap, zimap = sph2cart(qimap, thetamap, phimap)
        irr_nodes = np.zeros((len(ximap), 3))
        irr_nodes[:, 0] = ximap
        irr_nodes[:, 1] = yimap
        irr_nodes[:, 2] = zimap

        # Creating the regular grid with healpix
        # choose nside with L
        nside = L2nside(L)

        print('Creating Healpix grid with nside = '+str(nside))

        pix = hp.nside2npix(nside)
        listpix = np.arange(pix)
        vecx, vecy, vecz = hp.pix2vec(nside, listpix)

        mapx = q_shell*vecx
        mapy = q_shell*vecy
        mapz = q_shell*vecz

        nodes = np.zeros((len(mapx), 3))
        nodes[:, 0] = mapx
        nodes[:, 1] = mapy
        nodes[:, 2] = mapz

        # Inverse Distance Weighting (IDW) between tomographic and regular

        relative_pixel = hp.ang2pix(nside, thetamap, phimap)  # find the regular pix corresponding to the tomo
        neighbors_pixel = []
        nb_neighbors = np.zeros(pix)
        distances = []
        # inverse the map
        for pixel in range(pix):
            a = np.where(relative_pixel == pixel)
            if len(a[0]) == 0:
                raise ValueError('Interpolation failed. There is a hole')

            neighbors_pixel.append(a[0])
            nb_neighbors[pixel] = len(neighbors_pixel[pixel])
            d = np.sqrt((mapx[pixel]-ximap[a[0]])**2 + (mapy[pixel]-yimap[a[0]])**2+(mapz[pixel]-zimap[a[0]])**2)
            distances.append(d)

        ''' spherical EMC ALGORITHM '''

        M = len(listRj)
        # init if L ==3, use a random init. Else L>3, use the previous values
        # the coefficients.
        if L == 3:

            MPC = MPiik.mean()
            MIj = MPC*np.ones((Npix_dec, M)) + 0.4*MPC*(np.random.rand(Npix_dec, M)-1)

        else:
            Ip = hp.alm2map(Ilm, nside)
            neg_val = np.where(Ip < 0)
            Ip[neg_val] = 0
            VIj = hp.pixelfunc.get_interp_val(Ip, thetamap, phimap)
            MIj = VIj.reshape((M, Npix_dec)).T

        # heart of the algo
        grad = 1  # initialize gradient value

        while (grad > eta) or (nit_resol < nitmin):

            # Maximisation
            MPjk = compute_Pjk_matrix(MPiik, MIj, inds_k_non_zero, inds_k_zero, wj)
            # Compute Likelihood per pattern and total

            LikelihoodPattern = computeLikelihoodPattern(MPjk, MIj, MPiik, wj)
            Likelihoodn = LikelihoodPattern.mean()
            likelihood.append(Likelihoodn)
            print(Likelihoodn)

            MIj = np.zeros((Npix_dec, M))
            for n in range(Npix_dec):
                for j in range(M):
                    MIj[n, j] = (MPjk[j, inds_i_non_zero[n]]*MPiik[n, inds_i_non_zero[n]]).sum()

                    norm = sum(MPjk[j, :])

                    MIj[n, j] = MIj[n, j]/norm

            # compression
            VIj = MIj.T.ravel()
            Ip = np.zeros(pix)
            for pixel in range(pix):
                d = distances[pixel]
                w = 1.0 / d**2
                inds = neighbors_pixel[pixel]
                Ip[pixel] = np.sum(w*VIj[inds])/np.sum(w)

            Ilm = hp.map2alm(Ip, L-1)
            Ilm = friedel_sym(Ilm, L-1)

            i += 1
            # expansion
            Ip = hp.alm2map(Ilm, nside)
            neg_val = np.where(Ip < 0)
            Ip[neg_val] = 0
            VIj = hp.pixelfunc.get_interp_val(Ip, thetamap, phimap)
            MIj = VIj.reshape((M, Npix_dec)).T

            IlmArray[:np.size(Ilm), i-1] = Ilm

            LikelihoodArray[:, i-1] = LikelihoodPattern

            np.save('Ilm'+str(shellIndex)+'.npy', IlmArray)
            np.save('Likelihood'+str(shellIndex)+'.npy', LikelihoodArray)

            if i > 1:

                grad = np.abs((Likelihoodn - likelihood[i-2])/likelihood[i-2])

                print(grad)
            if i > 100:
                grad = eta*0.1
                print("exiting: too much iterations")

            nit_resol += 1

    return IlmArray, LikelihoodArray

# Example: Spherical EMC on shell 40, using 1000 diffraction patterns among
# those 10000 simulated, up to bandlimit Lmax = 17
IlmArray, LikelihoodArray = EMC_shell_coefficients(40, 10000, 1000, 17)
