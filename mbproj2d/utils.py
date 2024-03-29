# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Collection of useful functions."""

import math
import sys
import os
import time
import uuid
import hashlib
import pickle

import numpy as N
import h5py
from scipy.special import gammaln

import pyfftw
from pyfftw import zeros_aligned, ones_aligned, empty_aligned
pyfftw.interfaces.cache.enable()

from .physconstants import kpc_cm, ne_nH, Mpc_cm, kpc3_cm3

def uprint(*args, file=sys.stdout):
    """Unbuffered print."""
    print(*args, file=file)
    file.flush()

def diffQuart(a, b):
    """Calculate a**4-b**4."""
    return (a**2+b**2)*(a+b)*(a-b)

def diffCube(a, b):
    """Calculate a**3-b**3."""
    return (a-b)*(a*a+a*b+b*b)

def diffSqr(a, b):
    """Calculate a**2-b**2."""
    return (a+b)*(a-b)

def projectionVolume(R1, R2, y1, y2):
    """Return the projected volume of a shell of radius R1->R2 onto an
    annulus on the sky of y1->y2.

    this is the integral:
    Int(y=y1,y2) Int(x=sqrt(R1^2-y^2),sqrt(R2^2-y^2)) 2*pi*y dx dy
    =
    Int(y=y1,y2) 2*pi*y*( sqrt(R2^2-y^2) - sqrt(R1^2-y^2) ) dy

    This is half the total volume (front only)
    """

    def truncSqrt(x):
        return N.sqrt(N.clip(x, 0., 1e200))

    p1 = truncSqrt(R1**2 - y2**2)
    p2 = truncSqrt(R1**2 - y1**2)
    p3 = truncSqrt(R2**2 - y2**2)
    p4 = truncSqrt(R2**2 - y1**2)

    return (2/3*math.pi) * ((p1**3 - p2**3) + (p4**3 - p3**3))

def projectionVolumeMatrix(radii):
    """Calculate volumes (front and back) using a matrix calculation.

    Dot matrix with emissivity array to compute projected surface
    brightnesses.

    Output looks like this:
    >>> utils.projectionVolumeMatrix(N.arange(5))
    array([[  4.1887902 ,   7.55593906,   6.57110358,   6.4200197 ],
           [  0.        ,  21.76559237,  26.1838121 ,  21.27257712],
           [  0.        ,   0.        ,  46.83209821,  49.71516053],
           [  0.        ,   0.        ,   0.        ,  77.57748023]])

    """

    i_s, j_s = N.indices((len(radii)-1, len(radii)-1))

    radii_2 = radii**2
    y1_2 = radii_2[i_s]
    y2_2 = radii_2[i_s+1]
    R1_2 = radii_2[j_s]
    R2_2 = radii_2[j_s+1]

    p1 = (R1_2-y2_2).clip(0)
    p2 = (R1_2-y1_2).clip(0)
    p3 = (R2_2-y2_2).clip(0)
    p4 = (R2_2-y1_2).clip(0)

    return (4/3*math.pi) * ((p1**1.5 - p2**1.5) + (p4**1.5 - p3**1.5))

def calcNeSqdToNormPerKpc3(cosmo):
    """Compute factor to convert a ne-squared to a norm per kpc3."""
    return (
        1e-14 / (4*math.pi*(cosmo.D_A*Mpc_cm * (1.+cosmo.z))**2) /
        ne_nH * kpc3_cm3
    )

def symmetriseErrors(data):
    """Take numpy-format data,+,- and convert to data,+-."""
    symerr = N.sqrt( 0.5*(data[:,1]**2 + data[:,2]**2) )
    datacpy = N.array(data[:,0:2])
    datacpy[:,1] = symerr
    return datacpy

def calcMedianErrors(results):
    """Take a set of repeated results, and calculate the median and errors (from perecentiles)."""
    r = N.array(results)
    r.sort(0)
    num = r.shape[0]
    medians = r[ int(num*0.5) ]
    lowpcs = r[ int(num*0.1585) ]
    uppcs = r[ int(num*0.8415) ]

    return medians, uppcs-medians, lowpcs-medians

def calcChi2(model, data, error):
    """Calculate chi2 between model and data."""
    return (((data-model)/error)**2).sum()

def cashLogLikelihood(data, model):
    """Calculate log likelihood of Cash statistic."""

    like = N.sum(data * N.log(model)) - N.sum(model) - N.sum(gammaln(data+1))
    if N.isfinite(like):
        return like
    return -N.inf

class WithLock:
    """Hacky lockfile class."""

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        while True:
            try:
                os.mkdir(self.filename)
                break
            except OSError:
                time.sleep(1)

    def __exit__(self, type, value, traceback):
        os.rmdir(self.filename)

class AtomicWriteFile(object):
    """Write to a file, renaming to final name when finished."""

    def __init__(self, filename):
        self.filename = filename
        self.tempfilename = filename + '.temp' + str(uuid.uuid4())
        self.f = open(self.tempfilename, 'w')

    def __enter__(self):
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()
        os.rename(self.tempfilename, self.filename)

def gehrels(c):
    return 1. + N.sqrt(c + 0.75)

def run_rfft2(x):
    """Call numpy or pyfftw for rfft2."""
    return pyfftw.interfaces.numpy_fft.rfft2(x)

def run_irfft2(x):
    """Call numpy or pyfftw for irfft2."""
    return pyfftw.interfaces.numpy_fft.irfft2(x)

def loadChainFromFile(chainfname, pars, burn=0, thin=10, randsamples=None):
    """Get list of parameter values from chain.
    Walkers are collapsed into one dimension.
    Output dimensions are (numsamples, numpars).

    :param chainfname: input chain HDF5 file
    :param pars: Pars() object to check against chain parameters
    :param burn: how many iterations to remove off input
    :param thin: discard every N entries
    :param randsamples: randomly sample N samples if set (ignores thin)
    """

    with h5py.File(chainfname, 'r') as f:
        freekeys = pars.freeKeys()
        filefree = [x.decode('utf-8') for x in f['thawed_params']]
        if freekeys != filefree:
            raise RuntimeError('Parameters do not match those in chain')

        if randsamples is not None:
            chain = N.array(f['chain'][burn:, :, :])
            chain = chain.reshape(-1, chain.shape[2])
            rows = N.arange(chain.shape[0])
            N.random.shuffle(rows)
            chain = chain[rows[:randsamples], :]
        else:
            chain = N.array(f['chain'][burn::thin, :, :])
            chain = chain.reshape(-1, chain.shape[2])

    return chain

def binImage(img, factor, mean=False):
    """Bin up image by factor.

    :param img: input image array
    :param factor: integer bin factor for both dimensions
    :param mean: If True, calculate means, otherwise calculate sums
    """

    oyw, oxw = img.shape

    nyw = oyw//factor if oyw%factor==0 else oyw//factor+1
    nxw = oxw//factor if oxw%factor==0 else oxw//factor+1

    nimg = N.zeros((nyw, nxw), dtype=img.dtype)

    for dy in range(factor):
        for dx in range(factor):
            sel = img[dy::factor, dx::factor]
            nimg[:sel.shape[0], :sel.shape[1]] += sel

    if mean:
        cts = binImage(N.ones(img.shape, dtype=N.int32), factor)
        nimg = nimg / cts

    return nimg

class ConvPSFHelper:
    """Helper class for doing convolution of image with PSF.

    :param psf: input PSF array (2D image)
    """

    def __init__(self, psf):
        self.updatePSF(psf)

    def updatePSF(self, psf):
        """Set convolution code to use new PSF."""
        self.psf = psf

        imgshape = self.psf.shape
        self.in_realimg = pyfftw.zeros_aligned(imgshape, dtype=N.float32)
        self.in_realpsf = pyfftw.zeros_aligned(imgshape, dtype=N.float32)
        # output real->complex
        cshape = (imgshape[0], imgshape[1]//2+1)
        self.temp_cmplx = pyfftw.zeros_aligned(cshape, dtype=N.complex64)
        self.temp_cmplxpsf = pyfftw.zeros_aligned(cshape, dtype=N.complex64)
        self.out_real = pyfftw.zeros_aligned(imgshape, dtype=N.float32)

        self.fwd_fftimg = pyfftw.FFTW(
            self.in_realimg, self.temp_cmplx,
            axes=(0,1),
            direction='FFTW_FORWARD',
            flags=['FFTW_MEASURE'],
        )
        self.bkd_fftout = pyfftw.FFTW(
            self.temp_cmplx, self.out_real,
            axes=(0,1),
            direction='FFTW_BACKWARD',
            flags=['FFTW_MEASURE'],
        )

        # compute fft of psf (only used once)
        self.fwd_fftpsf = pyfftw.FFTW(
            self.in_realpsf, self.temp_cmplxpsf,
            axes=(0,1),
            direction='FFTW_FORWARD',
            flags=['FFTW_ESTIMATE'],
        )
        self.in_realpsf[:,:] = psf
        self.fwd_fftpsf()

    def doConv(self, inimg, outimg):
        """Do convolution with PSF.

        :param inimg: where to get input
        :param outimg: where to place output
        """

        self.fwd_fftimg.update_arrays(inimg, self.temp_cmplx)
        self.fwd_fftimg()
        self.temp_cmplx *= self.temp_cmplxpsf
        self.bkd_fftout.update_arrays(self.temp_cmplx, outimg)
        self.bkd_fftout()

    def __getstate__(self):
        """Get state for pickling.

        (this ensures alignment and plans are setup correctly)
        """
        return {'psf': self.psf}

    def __setstate__(self, state):
        """Set state after unpickling

        (this ensures alignment and plans are setup correctly)
        """
        self.updatePSF(state['psf'])

class CacheOnKey:
    """Cacheing class which writes to a pickled file in a subdirectory.

    Items are indexed via md5 hash.

    The first two characters are used to create a subdirectory within
    cachedir to reduce the number of files per directory.
    """

    cachedir = 'mbproj2d_cache'

    def __init__(self, key):
        self.key = key
        h = hashlib.md5()
        h.update(key.encode('utf8'))
        self.keyhash = h.hexdigest()
        self.subdir = os.path.join(self.cachedir, self.keyhash[:2])
        self.resultsfile = os.path.join(
            self.subdir, self.keyhash+'.pickle')
        self.lockfile = os.path.join(
            self.subdir, self.keyhash+'.lock')

    def __enter__(self):
        # make the output directory and subdirectory if required
        os.makedirs(self.subdir, exist_ok=True)

        # create the lockfile directory (wait if reqd)
        while True:
            try:
                os.mkdir(self.lockfile)
                break
            except OSError:
                time.sleep(1)
        return self

    def __exit__(self, type, value, traceback):
        # remove lock directory
        os.rmdir(self.lockfile)

    def exists(self):
        """Return whether cached value exists."""
        return os.path.exists(self.resultsfile)

    def read(self):
        """Read cached value, or returns KeyError if not found."""

        try:
            with open(self.resultsfile, 'rb') as fin:
                keyin, res = pickle.load(fin)
        except FileNotFoundError:
            raise KeyError('No such key to load')

        assert keyin == self.key
        return res

    def write(self, val):
        """Update or write the cached value."""

        with open(self.resultsfile, 'wb') as fout:
            pickle.dump(
                (self.key, val),
                fout,
                protocol=pickle.HIGHEST_PROTOCOL
            )
