MBProj2D
========

https://github.com/jeremysanders/mbproj2d

Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>

MBProj2D is released under the GNU Lesser General Public License
version 3 or greater. See the file COPYING for details.

More documentation can be found at http://mbproj2d.readthedocs.io/

Introduction
------------

MBProj2D (MultiBand Projector 2D) is a code which forward-models
images of galaxy clusters. Using a single X-ray band the model would
be sensitive to the density of the intracluster medium (ICM). By using
multiple bands the code is able to model the temperature variation
within the cluster. Given sufficient energy bands the metallicity of
the ICM can also be fitted.

MBProj2D can assume hydrostatic equilibrium using a mass model. From
this model (with an outer pressure parameter and the density profile)
the pressure profile can be computed for the cluster. This allows the
temperature to be computed on small spatial scales, which would
otherwise not have enough data to compute the temperature
independently. If no hydrostatic equilibrium is assumed then MBProj2D
can fit for the temperature of the ICM instead.

The model is normally first fit to the surface brightness images. MCMC
using the emcee module is used to compute a chain of model parameters,
from which posterior probability distributions can be computed for a
large number of model and physical parameters.


Requirements
------------

MBProj2D requires the following:

1. Python 3.3+ or greater
2. emcee  https://emcee.readthedocs.io/en/stable/ (Python module)
3. h5py   http://www.h5py.org/ (Python module)
4. numpy  https://www.numpy.org/ (Python module)
5. scipy  https://www.scipy.org/ (Python module)
6. astropy https://www.astropy.org/ (Python module)
7. pyfftw https://pypi.org/project/pyFFTW/ (Python module)
8. cython https://cython.org/
9. xspec  https://heasarc.gsfc.nasa.gov/xanadu/xspec/

Installation
------------

You can either install the module through its setup script:

::

  $ python setup.py build
  $ python setup.py install

or install via pip:

::

  $ pip install git+https://github.com/jeremysanders/mbproj2d.git


