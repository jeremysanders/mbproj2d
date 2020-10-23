MBProj2D
========

2D forward-fitting projection code for fitting images of clusters of
galaxies. This work builds upon the ideas in MBProj2.

This is *in progress* and is not yet suitable for serious use.

Copyright Jeremy Sanders (2020)

License: GPLv3

Requirements:
 - Recent Intel and AMD x86-64 processors with AVX2 support
 - python3
 - numpy
 - scipy
 - cython
 - pyfftw
 - h5py
 - emcee
 - astropy

Usage notes:

 - If using a PSF model, I suggest fitting a larger region of the sky than necessary, but masking out the edges. Do not zero the exposure map in these regions. The PSF modelling uses a FFT convolution, so the model will wrap around at the edges.
 
 - Input images must have even numbers of pixels on each axis
