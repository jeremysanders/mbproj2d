# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from .xspechelper import XSpecContext

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

    hdffname = 'mbproj2d_cache.hdf5'

    def __init__(self, cosmo, rmf, arf, emin_keV, emax_keV, NH_1022pcm2):

        if not os.path.exists(rmf):
            raise RuntimeError('RMF %s does not exist' % rmf)

        self.cosmo = cosmo
        self.rmf = rmf
        self.arf = arf
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.NH_1022pcm2 = NH_1022pcm2

        # build a key to lookup/store in the cache file
        h = hashlib.md5()
        h.update(os.path.abspath(rmf).encode('utf8'))
        h.update(os.path.abspath(arf).encode('utf8'))
        h.update(N.array((cosmo.z, emin_keV, emax_keV, NH_1022pcm2)))
        h.update(self.Tlogvals)
        self.key = 'rates_' + h.hexdigest()

        self._cacheRates()

    def _cacheRates(self):
        """Work out rate per kpc3 for temperature values for key given for Z=0,1
        """

        # lock hdf5 for concurrent access
        with utils.WithLock(self.hdffname + '.lockdir') as lock:
            with h5py.File(self.hdffname, 'a') as fcache:

                if self.key in fcache:
                    # already calculated
                    ZTrates = N.array(fcache[self.key])
                else:
                    # calculate using xspec
                    with XSpecContext() as xspec:
                        xspec.changeResponse(
                            self.rmf, self.arf, self.emin_keV, self.emax_keV)
                        ZTrates = N.array([
                            [
                                xspec.getCountsPerSec(
                                    self.NH_1022pcm2, T, Z, self.cosmo, 1.)
                                for T in N.exp(self.Tlogvals)
                            ]
                            for Z in (0, 1)
                        ])
                    fcache[self.key] = ZTrates
                    attrs = fcache[self.key].attrs
                    attrs['rmf'] = self.rmf
                    attrs['arf'] = self.arf
                    attrs['z'] = self.cosmo.z
                    attrs['NH_1022pcm2'] = self.NH_1022pcm2
                    attrs['erange_keV'] = (self.emin_keV, self.emax_keV)
                    attrs['Tlogvals'] = self.Tlogvals

        Ztrates = N.clip(ZTrates, 1e-100, None)
        self.Z0rates = N.log(ZTrates[0])
        self.Z1rates = N.log(ZTrates[1])

    def get(self, ne_pcm3, T_keV, Z_solar):
        """Get the count rates per kpc3 for an array of T,Z,ne."""

        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)
        logT = N.log(T_keV)
        Z0T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z0rates))
        Z1T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z1rates))

        # use Z=0 and Z=1 count rates to evaluate at Z given
        return (Z0T_ctrate + (Z1T_ctrate-Z0T_ctrate)*Z_solar) * ne_cm3**2

class FluxCalc:
    """Get fluxes for temperatures, densities and metallicities using xspec
    (results are erg/cm2/s/kpc3)

    Results are cached in a file for subsequent runs
    """

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)

    hdffname = 'mbproj2d_cache.hdf5'

    def __init__(self, cosmo, emin_keV, emax_keV, NH_1022pcm2=0):
        self.cosmo = cosmo
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.NH_1022pcm2 = NH_1022pcm2

        # build a key to lookup/store in the cache file
        h = hashlib.md5()
        h.update(N.array((cosmo.z, emin_keV, emax_keV, NH_1022pcm2)))
        h.update(self.Tlogvals)
        self.key = 'flux_' + h.hexdigest()

        self._cacheFluxes()

    def _cacheFluxes(self):
        """Work out fluxes per kpc3 for temperature values for key given for Z=0,1
        """

        # lock hdf5 for concurrent access
        with utils.WithLock(self.hdffname + '.lockdir') as lock:
            with h5py.File(self.hdffname, 'a') as fcache:

                if self.key in fcache:
                    # already calculated
                    ZTfluxes = N.array(fcache[self.key])
                else:
                    # calculate using xspec
                    with XSpecContext() as xspec:
                        xspec.dummyResponse()
                        ZTfluxes = N.array([
                            [
                                xspec.getFlux(
                                    self.NH_1022pcm2,
                                    T, Z, self.cosmo, 1.,
                                    emin_keV=self.emin_keV,
                                    emax_keV=self.emax_keV)
                                for T in N.exp(self.Tlogvals)
                            ]
                            for Z in (0, 1)
                        ])
                    fcache[self.key] = ZTfluxes
                    attrs = fcache[self.key].attrs
                    attrs['z'] = self.cosmo.z
                    attrs['NH_1022pcm2'] = self.NH_1022pcm2
                    attrs['erange_keV'] = (self.emin_keV, self.emax_keV)
                    attrs['Tlogvals'] = self.Tlogvals

        Ztfluxes = N.clip(ZTfluxes, 1e-100, None)
        self.Z0fluxes = N.log(ZTfluxes[0])
        self.Z1fluxes = N.log(ZTfluxes[1])

    def get(self, ne_pcm3, T_keV, Z_solar):
        """Get the fluxes per kpc3 for an array of T,Z,ne."""

        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)
        logT = N.log(T_keV)
        Z0T_flux = N.exp(N.interp(logT, self.Tlogvals, self.Z0fluxes))
        Z1T_flux = N.exp(N.interp(logT, self.Tlogvals, self.Z1fluxes))

        # use Z=0 and Z=1 count fluxes to evaluate at Z given
        return (Z0T_flux + (Z1T_flux-Z0T_flux)*Z_solar) * ne_pcm3**2
