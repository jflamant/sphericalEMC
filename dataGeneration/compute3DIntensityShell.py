#   file: compute3DIntensityShell.py
#
#   This code computes theoretical intensity functions on shells of specified
#   index. It extracts the structural information contained in a PDB file, then
#   computes the theoretical intensity on each shell given tabulated X-ray
#   scattering factors. Shell intensities are saved as separate NumPy arrays.
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
from Bio.PDB import *  # To parse PDB files
import periodictable  # Periodic table
from periodictable import xsf  # X-ray scattering factors
from math import pi


def cart2sph(xi, yi, zi):

    qi = np.sqrt(xi**2+yi**2+zi**2)
    theta = np.arccos(zi/qi)
    phi = np.arctan2(yi, xi)

    return qi, theta, phi


# parameters

wavelength = 2  # in Angstroms
D = 1e-2  # detector distance, in meters
px = 3.38e-5  # pixel size, in meters
re = 2.8179403267*1e-15  # in m
nphotons = 1e13
beamarea = 1e-14  # in m**2

qmax = 4*pi/wavelength  # largest scattering vector
grid_size = 64  # number of pixels in the upper half detector

# largest scattering vector available on the detector
qmax_grid = qmax*np.sin(0.5*np.arctan(px*grid_size/D))

# define spherical resolution of the intensity with nside
nside = 128
shell_index = np.arange(1, grid_size+1, 1)  # an index for shells

# array of q values, the detector is supposed to be a square matrix
radial_seq = qmax*np.sin(0.5*np.arctan(px*shell_index/D))

# Creation of the shell grid

pix = hp.nside2npix(nside)
listpix = np.arange(pix)
vecx, vecy, vecz = hp.pix2vec(nside, listpix)  # points on the unit sphere

# define the 3D grid, given radial_seq (radial points)
for shell in range(1, grid_size+1):
    mapx = radial_seq[shell-1]*vecx
    mapy = radial_seq[shell-1]*vecy
    mapz = radial_seq[shell-1]*vecz
    q, theta, phi = cart2sph(mapx, mapy, mapz)

    # compute the shell intensity

    parser = PDBParser()

    structure = parser.get_structure("2BUK", "./pdb/2BUK.pdb")

    gatoms = structure.get_atoms()
    atoms = []
    for at in gatoms:
        atoms.append(at)

    normq = np.sqrt(mapx**2 + mapy**2 + mapz**2)

    F = np.zeros(len(mapx), dtype=complex)

    nat = 0
    for at in atoms:
        nat += 1.0
        percent = round(nat/len(atoms)*100, 2)
        if percent % 1 == 0:
            print('Computing: ' + str(percent) + ' % ')

        el = periodictable.elements.symbol(at.element)
        pos = at.get_coord()
        Xdata = xsf.Xray(el)
        Qpos = mapx*pos[0] + mapy*pos[1] + mapz*pos[2]
        F += Xdata.f0(normq)*np.exp(1j*Qpos)

    I = abs(F)**2*re**2*nphotons/beamarea*(px/D)**2
    np.save('./shells/I_2BUK_'+str(shell)+'.npy', I)  # save computed intensity
