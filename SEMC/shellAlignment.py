#   file: shellAlignment.py
#
#   This code realigns the differents shells obtained via the spherical EMC
#   EMC algorithm. Namely, it computes the cross-correlation on SO(3) between
#   two successive shells thanks to their spherical harmonic representation.
#
#   See also: shellAlignmentLibrary.py
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
import matplotlib.pyplot as plt
import healpy as hp
import os
from shellAlignmentLibrary import *



def L2nside(L):

    ''' Gives the closest nside corresponding to L, such that L <=2nside+1'''

    a = np.log((L-1.0)/2)/np.log(2)

    nside = int(2**(np.ceil(a)))

    return nside


def cleanifyCoeff(IlmArray, Lmax=7):

    coeffmax = hp.Alm.getsize(Lmax-1)

    lastit = np.nonzero(IlmArray[0, :])[0].max()
    for it in range(lastit + 1):
        sizeIlm = np.max(np.nonzero(IlmArray[:, it])[0])
        if sizeIlm + 1 == coeffmax:
            correctIt = it

    IlmOut = IlmArray[:coeffmax, correctIt]

    return IlmOut


def getReconstructionsAligned(idShell, Ilm_na, Lgrid, Lmax=7, vmin = 1e-4, nside=128):

    Ir = np.zeros((hp.nside2npix(nside), np.size(idShell)))
    for id, Ilm in enumerate(Ilm_na):
        Ilm = np.ascontiguousarray(Ilm)

        if id == 0:
            '''align the first shell with theoretically calculated intensity'''
            Ith = np.load('first_shell_to_start_alignment.npy')

            almth = hp.map2alm(Ith, Lmax-1)

            almr = alignShells(almth, Ilm, Lmax-1, Lgrid)

            rI = hp.alm2map(almr, nside)
            neg = np.where(rI < 0)
            rI[neg] = vmin
            Ir[:, 0] = rI

        else:
            ''' Align the succesive shells with respect to each preceding shell
            '''

            almr = alignShells(almr, Ilm, Lmax-1, Lgrid)
            rI = hp.alm2map(almr, nside)
            neg = np.where(rI < 0)
            rI[neg] = vmin
            Ir[:, id] = rI

    return Ir


# choose the shells and parameters
idShell = np.arange(5, 41)
Lmax = 7
Lgrid = 21  # controls the refinment of the SO3 grid
# load the data and clean
path = os.getcwd()

Ilm_na = np.zeros((np.size(idShell), hp.Alm.getsize(Lmax-1)), dtype=complex)
for id, s in enumerate(idShell):
    # load estimated Ilm vectors
    IlmArray = np.load(os.path.join(path, 'data', 'Ilm'+str(s)+'.npy'))
    Ilm_na[id, :] = cleanifyCoeff(IlmArray, Lmax=Lmax)

# align shells
Ir = getReconstructionsAligned(idShell, Ilm_na, Lgrid, Lmax=Lmax, vmin=1e-4, nside=128)
np.save('IrL'+str(Lmax)+'.npy', Ir)
plt.show()
