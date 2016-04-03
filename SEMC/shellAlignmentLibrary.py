#   file: shellAlignmentLibrary.py
#
#   This code contains several handy functions to perform the realignment
#   of consecutive shells.
#
#   See also: shellAlignment.py
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

from math import pi, floor
from scipy.special import jv, legendre, sph_harm, jacobi
from scipy.misc import comb

# GENERAL USE #######################


def wignerd(j, m, n=0, approx_lim=100):

    '''
        Wigner "small d" matrix. (Euler z-y-z convention)
        example:
            j = 2
            m = 1
            n = 0
            beta = linspace(0,pi,100)
            wd210 = wignerd(j,m,n)(beta)

        some conditions have to be met:
             j >= 0
            -j <= m <= j
            -j <= n <= j

        The approx_lim determines at what point
        bessel functions are used. Default is when:
            j > m+10
              and
            j > n+10

        for integer l and n=0, we can use the spherical harmonics. If in
        addition m=0, we can use the ordinary legendre polynomials.
    '''

    if (j < 0) or (abs(m) > j) or (abs(n) > j):
        raise ValueError("wignerd(j = {0}, m = {1}, n = {2}) value error.".format(j,m,n) \
            + " Valid range for parameters: j>=0, -j<=m,n<=j.")

    if (j > (m + approx_lim)) and (j > (n + approx_lim)):
        #print 'bessel (approximation)'
        return lambda beta: jv(m-n, j*beta)

    if (floor(j) == j) and (n == 0):
        if m == 0:
            #print 'legendre (exact)'
            return lambda beta: legendre(j)(np.cos(beta))
        elif False:
            #print 'spherical harmonics (exact)'
            a = np.sqrt(4.*pi / (2.*j + 1.))
            return lambda beta: a * np.conjugate(sph_harm(m,j,beta,0.))

    jmn_terms = {
        j+n : (m-n,m-n),
        j-n : (n-m,0.),
        j+m : (n-m,0.),
        j-m : (m-n,m-n),
        }

    k = min(jmn_terms)
    a, lmb = jmn_terms[k]

    b = 2.*j - 2.*k - a

    if (a < 0) or (b < 0):
        raise ValueError("wignerd(j = {0}, m = {1}, n = {2}) value error.".format(j,m,n) \
            + " Encountered negative values in (a,b) = ({0},{1})".format(a,b))

    coeff = np.power(-1.,lmb) * np.sqrt(comb(2.*j-k,k+a)) * (1./np.sqrt(comb(k+b,b)))
    
    
    #print 'jacobi (exact)'
    return lambda beta: coeff\
        * np.power(np.sin(0.5*beta),a) \
        * np.power(np.cos(0.5*beta),b) \
        * jacobi(k,a,b)(np.cos(beta))

    
def wignerD(j,m,n):
    '''
        Wigner D-function. (Euler z-y-z convention)

        This returns a function of 2 to 3 Euler angles:
            (alpha, beta, gamma)

        gamma defaults to zero and does not need to be
        specified.

        The approx_lim determines at what point
        bessel functions are used. Default is when:
            j > m+10
              and
            j > n+10

        usage:
            from numpy import linspace, meshgrid
            a = linspace(0, 2*pi, 100)
            b = linspace(0,   pi, 100)
            aa,bb = meshgrid(a,b)
            j,m,n = 1,1,1
            zz = wignerD(j,m,n)(aa,bb)
    '''

    return lambda alpha,beta,gamma: \
          np.exp(-1j*m*alpha) \
        * wignerd(j,m,n)(beta) \
        * np.exp(-1j*n*gamma)


# FFT on the rotation group using the material from Kostelec and Rockmore

def sampling_grid(L):
    #sampling grid to perform FFT on SO3 for a L-bandlimited function.
    #It is given in zyz-Euler angles, (alpha,beta,gamma)
    
    id_val = np.arange(0,2*L,1)
    alpha = 2*pi/(2*L)*id_val
    beta = pi/(4*L)*(2*id_val+1)
    gamma = 2*pi/(2*L)*id_val
    
    euler = np.zeros((8*L**3,3))
    i=0
    for j1 in range(2*L):
        for k in range(2*L):
            for j2 in range(2*L):
                euler[i,0] = alpha[j1]
                euler[i,1] = beta[k]
                euler[i,2] = gamma[j2]
                i+=1
    return euler
##############################
# conversion between healpix and traditional l,m
def hp2lm(alm,l,m,lmax):

	if m >= 0:
		id = hp.Alm.getidx(lmax,l,m)
		blm = alm[id]
	else:
		id = hp.Alm.getidx(lmax,l,-m)
		blm = (-1)**(-m)*np.conj(alm[id])
	return blm
 #rotate alm's (assumed to be in healpix standard)
def rotatealm(alm,lmax,alpha,beta,gamma):
    
    blm = np.zeros((lmax+1)**2,dtype=complex)
    i=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            for n in range(-l,l+1):
                alm_t = hp2lm(alm,l,n,lmax)
                blm[i] += alm_t*wignerD(l,m,n)(alpha,beta,gamma)
            i+=1
            
    return blm
    
#conversion bteween standard et healpix format
def stand2hp(alm,lmax):
    size = hp.Alm.getsize(lmax)
    blm = np.zeros(size,dtype=complex)
    i=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            
            if m>=0:
                id = hp.Alm.getidx(lmax,l,m)
                blm[id] = alm[i]
            i+=1
    return blm

def alignShells(alm, blm, lmax, Lgrid):

    L = lmax+1
    euler = sampling_grid(Lgrid)
    N = np.size(euler, 0)

    Corr = np.zeros(N, dtype=complex)
    for l in range(L):
        for m in range(-l,l+1):
            for n in range(-l,l+1):
                dlmn = wignerD(l,m,n)(euler[:,0],euler[:,1],euler[:,2])

                a = hp2lm(alm, l, m, lmax)
                b = hp2lm(blm,l,n,lmax)
                Corr+= a*np.conj(b)*np.conj(dlmn)

    x = np.where(Corr==max(Corr))
    x = x[0]

    # now we rotate back blm         
    blmr = rotatealm(blm,lmax,euler[x,0][0],euler[x,1][0],euler[x,2][0])

    hpblmr = stand2hp(blmr,lmax)

    return hpblmr
