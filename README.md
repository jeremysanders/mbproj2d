MBProj2D
========

2D forward-fitting projection code for fitting images of clusters of
galaxies. This work builds upon the ideas in MBProj2 (see https://github.com/jeremysanders/mbproj2 and https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.1065S). It differs from MBProj2 in that it fits images, rather than profiles, and allows multiple cluster and background components to be simultaneously fitted.

This code is *in development*, so please be aware that the interface can be unstable. More documentation can be found at https://mbproj2d.readthedocs.io/en/latest/

Copyright Jeremy Sanders (2020)

License: LGPLv3

Requirements:
 - Xspec (https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/)
 - python3
 - numpy
 - scipy
 - cython
 - pyfftw
 - h5py
 - emcee
 - astropy
 - nlopt (optional, for better fitting)

Installation instructions using pip:

    python3 -m pip install astropy cython emcee h5py numpy pyfftw scipy
    python3 -m pip install git+https://github.com/jeremysanders/mbproj2d

If using conda, the following package is also required for Linux:

    conda install -c conda-forge gxx_linux-64

Usage notes:

 - If using a PSF model, I suggest fitting a larger region of the sky than necessary, but masking out the edges. Do not zero the exposure map in these regions. The PSF modelling uses a FFT convolution, so the model will wrap around at the edges. Please note that the `pad=N` option in `Image()` or `imageLoad` will automatically zero-pad images with `N` masked-out pixels.

 - Input images must have even numbers of pixels on each axis if optimal_size not set
