# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the Free
# Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
# MA 02111-1307, USA

"""Module to get count rates for temperatures, densities and
metallicities.

Results are taken from xspec, interpolating between results at fixed
temperatures and metallicities
"""

import os.path
import hashlib

import h5py
import numpy as N
import scipy.interpolate

from . import utils
from .xspechelper import XSpecHelper

class RateCalc:
    """Get count rates for temperatures, densities and metallicities using xspec

    This calculates results by getting the rates for Z=0 and 1, then interpolating
    The rates are calculated for a grid of temperature measurements.

    Results are cached in a file for subsequent runs
    """

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)

    hdffname = 'countrate_cache_2d.hdf5'
    
    def __init__(self, cosmo, rmf, arf, minenergy_keV, maxenergy_keV, NH_1022pcm2):

        if not os.path.exists(rmf):
            raise RuntimeError('RMF %s does not exist' % rmf)

        self.cosmo = cosmo
        self.rmf = rmf
        self.arf = arf
        self.minenergy_keV
        self.maxenergy_keV
        self.NH_1022pcm2

        # build a key to lookup/store in the cache file
        h = hashlib.md5()
        h.update(rmf)
        h.update(arf)
        h.update(N.array(cosmo.z, minenergy_keV, maxenergy_keV, NH_1022pcm2))
        h.update(self.Tlogvals)
        self.key = h.hexdigest()

        self._cacheRates()

    def _cacheRates(self):
        """Work out rate for temperature values for key given for Z=0,1
        """

        # lock hdf5 for concurrent access
        with utils.WithLock(self.hdffname + '.lockdir') as lock:
            with h5py.File(self.hdffname) as fcache:

                if self.key in fcache:
                    # already calculated
                    ZTrates = N.array(fcache[self.key])
                else:
                    # calculate using xspec                    
                    with XspecContext() as xspec:
                        xspec.changeResponse(
                            self.rmf, self.arf, self.minenergy_keV, self.maxenergy_keV)
                        ZTrates = N.array([
                            [
                                xspec.getCountsPerSec(
                                    self.NH_1022pcm2, T, Z, self.cosmo.z, 1.)
                                for T in N.exp(self.Tlogvals)
                            ]
                            for Z in (0, 1)
                        ])
                    fcache[self.key] = ZTrates

        Ztrates = N.clip(Ztrates, 1e-100, None)
        self.Z0rates = N.log(ZTrates[0])
        self.Z1rates = N.log(ZTrates[1])

    def getRate(T_keV, Z_solar, ne_cm3):
        """Get the count rates for an array of T,Z,ne."""
        
        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)
        logT = N.log(T_keV)
        Z0T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z0rates))
        Z1T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z1rates))
        
        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0T_ctrate + (Z1T_ctrate-Z0T_ctrate)*Z_solar) * ne_cm3**2
        
class FluxCalc:
    """Get unabsorbed flux for model with properties given."""
    
    def __init__(self):
        self.fluxcache = {}
    
    def getFlux(self, T_keV, Z_solar, ne_cm3, emin_keV=0.01, emax_keV=100.):
        """Get flux per cm3 in erg/cm2/s.

        emin_keV and emax_keV are the energy range
        """

        if (emin_keV, emax_keV) not in self.fluxcache:
            self.makeFluxCache(emin_keV, emax_keV)
        fluxcache = self.fluxcache[(emin_keV, emax_keV)]

        if not self.fluxcache:
            self.makeFluxCache()

        logT = N.log( N.clip(T_keV, self.Tmin, self.Tmax) )

        # evaluate interpolation functions for temperature given
        Z0_flux, Z1_flux = [f(logT) for f in fluxcache]

        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0_flux + (Z1_flux-Z0_flux)*Z_solar)*ne_cm3**2

    def makeFluxCache(self, emin_keV, emax_keV):
        """Work out fluxes for the temperature grid points and response."""

        with XspecContext() as xspec:
            xspec.dummyResponse()
            results = []
            # we can work out the counts at other metallicities from two values
            # we also work at a density of 1 cm^-3
            for Z_solar in (0., 1.):
                Zresults = []
                for Tlog in self.Tlogvals:
                    flux = xspec.getFlux(
                        N.exp(Tlog), Z_solar, self.cosmo, 1.,
                        emin_keV=emin_keV, emax_keV=emax_keV)
                    Zresults.append(flux)

        # store functions which interpolate the results from above
        results.append( scipy.interpolate.interpolate.interp1d(
            self.Tlogvals, N.array(Zresults), kind='cubic' ) )

        self.fluxcache[(emin_keV, emax_keV)] = tuple(results)
