# spherical EMC

A Python implementation of the spherical Expansion-Maximization-Compression algorithm -- An algorithm for intensity reconstruction in single-particle imaging experiments

The code comes in addition of the paper 
> Flamant, J., Bihan, N.L., Martin, A.V. and Manton, J.H., 2016. A spherical harmonic approach to single particle imaging with X-ray lasers.

available at http://arxiv.org/abs/1602.01301

## Requirements

Code has been tested for Python 3.5, but it should work for any Python 3.x version. Several packages are required:

- [NumPy](http://www.numpy.org)
- [SciPy](https://www.scipy.org)
- [Matplotlib](http://matplotlib.org)
- [Healpy](https://github.com/healpy/healpy)
- [periodictable 1.4.1](https://pypi.python.org/pypi/periodictable): X-ray scattering factors database
- [BioPython](http://biopython.org/wiki/Main_Page): Especially the module bio.pdb, to parse PDB files


## Overview

This repository contains several Python codes to simulate diffraction patterns, and reconstruct the 3D intensity function using the shell-by-shell approach developed in the paper. 

- **dataGeneration/** contains routines to generate *diffraction patterns* on a specified spherical shell. 
- **requiredPatterns/** contains a script to estimate the required number of diffraction patterns to reconstruct the intensity.
- **sphericalEMC/** contains the spherical EMC code, as well as the shell realignment procedures. 

## Usage example
1. Select one shell index, and compute the theoretical shell intensity using *compute3DIntensityShell.py*
2. Use *generateSamplesShell.py* to generate nbSamples on this shell, that is nbSamples 'diffraction patterns' on this shell. (These are Poisson realization of the intensity at randomly rotated versions of the detector reference sampling points)
3. Use *sphericalEMC.py* to reconstruct the shell intensity, up to some specified bandlimit $L$.
4. Iterate this procedure (which can be parallelized) for successive shells, and realign the shells using *shellAlignment.py*.

