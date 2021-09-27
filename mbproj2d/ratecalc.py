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

"""Module to get count rates for different spectral models.

Results are taken from xspec, interpolating between results and
caching between runs.

"""

import os.path

import h5py
import numpy as N
import scipy.interpolate

from . import utils
from .xspechelper import XSpecContext

class ApecRateCalc:
    """Get count rates for temperatures, norm and metallicities using xspec

    This calculates results by getting the rates for Z=0 and 1, then interpolating
    The rates are calculated for a grid of temperature measurements.

    Results are cached in a file for subsequent runs
    """

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)
    vers = 2.0

    abund = 'lodd'

    def __init__(self, rmf, arf, emin_keV, emax_keV, NH_1022pcm2, redshift):

        if not os.path.exists(rmf):
            raise RuntimeError('RMF %s does not exist' % rmf)

        self.rmf = rmf
        self.arf = arf
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.NH_1022pcm2 = NH_1022pcm2
        self.redshift = redshift

        # build a key to lookup/store in the cache file
        self.key = '\n'.join([
            'rates_apec',
            str(self.vers),
            rmf,
            arf,
            self.abund,
            str(redshift), str(emin_keV), str(emax_keV),
            str(NH_1022pcm2),
            repr(self.Tlogvals.tolist()),
        ])

        # for cacheing results
        with utils.CacheOnKey(self.key) as cache:
            if cache.exists():
                ZTrates = cache.read()
            else:
                # calculate using xspec
                with XSpecContext() as xspec:
                    xspec.changeResponse(
                        self.rmf, self.arf, self.emin_keV, self.emax_keV)
                    xspec.setAbund(self.abund)
                    # get rates for Z=0 and Z=1
                    Z0Trates, Z1Trates = [], []
                    for T in N.exp(self.Tlogvals):
                        xspec.setApec(self.NH_1022pcm2, T, 0, self.redshift, 1)
                        Z0Trates.append(xspec.getRate())
                        xspec.setApec(self.NH_1022pcm2, T, 1, self.redshift, 1)
                        Z1Trates.append(xspec.getRate())

                ZTrates = N.array([Z0Trates, Z1Trates])
                cache.write(ZTrates)

        Ztrates = N.clip(ZTrates, 1e-100, None)
        self.Z0rates = N.log(ZTrates[0])
        self.Z1rates = N.log(ZTrates[1])

    def get(self, T_keV, Z_solar, norm):
        """Get the count rates for an array of T,Z,norm

        If T is out of range, Z<0 or norm<0, return nan values
        """

        logT = N.log(T_keV)
        Z0T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z0rates))
        Z1T_ctrate = N.exp(N.interp(logT, self.Tlogvals, self.Z1rates))

        # use Z=0 and Z=1 count rates to evaluate at Z given
        rates = N.where(
            (T_keV >= self.Tmin) & (T_keV <= self.Tmax) &
            (Z_solar >= 0) & (norm >= 0),
            (Z0T_ctrate + (Z1T_ctrate-Z0T_ctrate)*Z_solar) * norm,
            N.nan)

        return rates

class PowerlawRateCalc:
    """Get fluxes in bands for a powerlaw models."""

    gamma_min = 0.0
    gamma_max = 3.0
    gamma_bins = 61
    gammas = N.linspace(gamma_min, gamma_max, gamma_bins)
    vers = 1.0

    def __init__(self, rmf, arf, emin_keV, emax_keV, NH_1022pcm2):

        # build a key to lookup/store in the cache file
        self.key = '\n'.join([
            'rates_plaw',
            str(self.vers),
            rmf,
            arf,
            str(emin_keV), str(emax_keV),
            str(NH_1022pcm2),
            repr(self.gammas.tolist()),
        ])

        # see whether calculations are in file
        with utils.CacheOnKey(self.key) as cache:
            if cache.exists():
                # already calculated
                self.rates = cache.read()
            else:
                # calculate
                rates = []
                with XSpecContext() as xspec:
                    xspec.changeResponse(rmf, arf, emin_keV, emax_keV)
                    for gamma in self.gammas:
                        xspec.setPlaw(NH_1022pcm2, gamma, 1.0)
                        rates.append(xspec.getRate())
                self.rates = N.array(rates)
                cache.write(self.rates)

    def get(self, gamma, norm):
        """Get powerlaw rate for a particular gamma and norm."""
        return N.interp(gamma, self.gammas, self.rates) * norm

class ApecFluxCalc:
    """Get fluxes for temperatures, densities and norms using xspec
    (results are erg/cm2/s)

    Results are cached in a file for subsequent runs
    """

    Tmin = 0.06
    Tmax = 60.
    Tsteps = 100
    Tlogvals = N.linspace(N.log(Tmin), N.log(Tmax), Tsteps)
    vers = 2.0

    abund = 'lodd'

    def __init__(self, emin_keV, emax_keV, redshift=0, NH_1022pcm2=0):
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.NH_1022pcm2 = NH_1022pcm2
        self.redshift = redshift

        # build a key to lookup/store in the cache file
        self.key = '\n'.join([
            'flux_apec',
            str(self.vers),
            self.abund,
            str(redshift), str(emin_keV), str(emax_keV),
            str(NH_1022pcm2),
            repr(self.Tlogvals.tolist()),
        ])

        with utils.CacheOnKey(self.key) as cache:
            if cache.exists():
                ZTfluxes = cache.read()
            else:
                # calculate using xspec
                with XSpecContext() as xspec:
                    xspec.dummyResponse()
                    xspec.setAbund(self.abund)
                    Z0fluxes, Z1fluxes = [], []
                    for T in N.exp(self.Tlogvals):
                        xspec.setApec(self.NH_1022pcm2, T, 0, self.redshift, 1)
                        Z0fluxes.append(xspec.getFlux(self.emin_keV, self.emax_keV))
                        xspec.setApec(self.NH_1022pcm2, T, 1, self.redshift, 1)
                        Z1fluxes.append(xspec.getFlux(self.emin_keV, self.emax_keV))

                ZTfluxes = N.array([Z0fluxes, Z1fluxes])
                cache.write(ZTfluxes)

        Ztfluxes = N.clip(ZTfluxes, 1e-100, None)
        self.Z0fluxes = N.log(ZTfluxes[0])
        self.Z1fluxes = N.log(ZTfluxes[1])

    def get(self, T_keV, Z_solar, norm):
        """Get the fluxes for an array of T, Z, norm."""

        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)
        logT = N.log(T_keV)
        Z0T_flux = N.exp(N.interp(logT, self.Tlogvals, self.Z0fluxes))
        Z1T_flux = N.exp(N.interp(logT, self.Tlogvals, self.Z1fluxes))

        # use Z=0 and Z=1 count fluxes to evaluate at Z given
        return (Z0T_flux + (Z1T_flux-Z0T_flux)*Z_solar) * norm

class XspecModelRateCalc:
    """Get rates for an xspec model.

    Returns rate in band for a model xcm file
    """

    vers = 2.0

    def __init__(self, rmf, arf, emin_keV, emax_keV, xcmfile):

        # build a key to lookup/store in the cache file
        # unclear whether we should bother caching...

        with open(xcmfile) as fin:
            self.key = '\n'.join([
                'xcm_file',
                str(self.vers),
                rmf,
                arf,
                str(emin_keV), str(emax_keV),
                fin.read(),
            ])

        with utils.CacheOnKey(self.key) as cache:
            if cache.exists():
                self.rate = cache.read()
            else:
                with XSpecContext() as xspec:
                    xspec.changeResponse(rmf, arf, emin_keV, emax_keV)
                    xspec.loadXCM(xcmfile)
                    self.rate = xspec.getRate()
                cache.write(self.rate)

    def get(self):
        return self.rate
