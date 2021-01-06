# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
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

import math
from collections import defaultdict
import numpy as N
import h5py

from .profile import Radii
from . import utils
from .utils import uprint
from .ratecalc import ApecFluxCalc
from .physconstants import (
    kpc_cm, keV_erg, ne_nH, mu_g, mu_e, boltzmann_erg_K, keV_K, Mpc_cm,
    yr_s, solar_mass_g, G_cgs, P_keV_to_erg, km_cm, kpc3_cm3)

# we want to define the cumulative values half way in the
# shell, so we have to split the luminosity and mass across
# the shell
def fracMassHalf(snum, edges):
    """Fraction of mass of shell which is in the inner and outer
    halves of (r1+r2)/2."""

    r1, r2 = edges[snum], edges[snum+1]
    # Integrate[4*Pi*r^2, {r, r1, (r1 + r2)/2}]
    #  (Pi*(-8*r1**3 + (r1 + r2)**3))/6.
    # Integrate[4*Pi*r^2, {r, (r1 + r2)/2, r2}]
    #  4*Pi*(r2**3/3. - (r1 + r2)**3/24.)
    volinside = (math.pi * (-8*r1**3 + (r1 + r2)**3))/6
    voloutside = 4*math.pi * (r2**3/3 - (r1 + r2)**3/24)
    finside = volinside / (volinside + voloutside)
    foutside = 1 - finside
    return finside, foutside

class Phys:
    """Calculate physical profiles for a cluster model.

    :param pars: parameters to apply
    :param model: Cluster model component (not TotalModel!)
    :param rmin_kpc: minimum radius to compute profiles for
    :param rmax_kpc: maximum radius to compute profiles for
    :param rsteps: number of steps to compute profiles for
    :param binning: should be 'log' or 'lin' for bin size
    :param average: should be 'midpt', 'volume', 'mean' for how to convert from input to output bins
    :param linbin_kpc: internal linear bin size to use before rebinning
    :param fluxrange_keV: energy range to compute (unabsorbed) fluxes between
    :param luminrange_keV: rest energy range to compute luminosities between
    """

    def __init__(self, pars, model,
                 rmin_kpc=0.5, rmax_kpc=2000, rsteps=256,
                 linbin_kpc=0.5,
                 binning='log',
                 average='midpt',
                 fluxrange_keV=(0.5,2.0),
                 luminrange_keV=(0.5,2.0),
    ):
        self.pars = pars
        self.model = model

        self.radii = Radii(linbin_kpc, int(rmax_kpc/linbin_kpc)+1)

        # for getting absorbed fluxes
        self.fluxrange = fluxrange_keV
        self.fluxcalc = ApecFluxCalc(
            fluxrange_keV[0], fluxrange_keV[1],
            NH_1022pcm2=model.NH_1022pcm2, redshift=model.cosmo.z)
        # for getting unabsorbed fluxes in rest band
        self.luminrange = luminrange_keV
        self.fluxcalc_lumin = ApecFluxCalc(
            luminrange_keV[0]/(1+model.cosmo.z),
            luminrange_keV[1]/(1+model.cosmo.z),
            redshift=model.cosmo.z,
            NH_1022pcm2=0)
        self.fluxcalc_lumin_bolo = ApecFluxCalc(
            0.01, 100, NH_1022pcm2=0, redshift=model.cosmo.z)

        # factor to convert from ne**2 -> norm/kpc3
        self.nesqd_to_norm = utils.calcNeSqdToNormPerKpc3(model.cosmo)

        # conversion factor from above to luminosity
        self.flux_to_lumin = 4*math.pi*(model.cosmo.D_L*Mpc_cm)**2

        if binning == 'log':
            self.out_edges_kpc = N.logspace(
                N.log10(rmin_kpc), N.log10(rmax_kpc), rsteps+1)
        elif binning == 'lin':
            self.out_edges_kpc = N.linspace(rmin_kpc, rmax_kpc, rsteps+1)
        else:
            raise RuntimeError('Invalid binning')

        self.out_centre_kpc = 0.5*(self.out_edges_kpc[:-1]+self.out_edges_kpc[1:])
        self.out_centre_cm = self.out_centre_kpc * kpc_cm
        self.out_vol_kpc3 = (4/3*math.pi)*utils.diffCube(
            self.out_edges_kpc[1:],self.out_edges_kpc[:-1])
        self.out_vol_cm3 = self.out_vol_kpc3 * kpc3_cm3

        self.out_projmatrix = utils.projectionVolumeMatrix(
            self.out_edges_kpc) * kpc3_cm3

        # calculate function to go from linearily binned profiles to
        # output profiles, depending on the average method chosen
        binidxs = N.searchsorted(self.out_edges_kpc, self.radii.cent_kpc)
        if average == 'midpt':
            self.rebinfn = lambda x: N.interp(
                self.out_centre_kpc, self.radii.cent_kpc, x)
        elif average == 'volume':
            invrebinvol = 1/N.bincount(binidxs, weights=self.radii.vol_kpc3)
            self.rebinfn = lambda x: (
                N.bincount(binidxs, weights=x*self.radii.vol_kpc3) * invrebinvol )
        elif average == 'mean':
            invcts = 1/N.bincount(binidxs)
            self.rebinfn = lambda x: N.bincts(binidxs, weights=x) * invcts
        else:
            raise RuntimeError('Invalid averaging mode')

    def calc(self, ne_pcm3, T_keV, Z_solar, g_cmps2, phi_ergpg):
        """Given input profiles, calculate output profiles.

        These are assumed to have the output radii spacing
        """

        v = {}
        nshells = len(ne_pcm3)
        v['ne_pcm3'] = ne_pcm3
        v['T_keV'] = T_keV
        v['Z_solar'] = Z_solar
        v['g_cmps2'] = g_cmps2
        v['potential_ergpg'] = phi_ergpg

        v['P_keVpcm3'] = ne_pcm3 * T_keV
        v['P_ergpcm3'] = T_keV * ne_pcm3 * P_keV_to_erg
        v['S_keVcm2'] = T_keV * ne_pcm3**(-2/3)
        v['vol_cm3'] = self.out_vol_cm3

        Mgas_g = ne_pcm3 * self.out_vol_cm3 * mu_e*mu_g
        v['Mgas_Msun'] = Mgas_g * (1/solar_mass_g)

        norm_pkpc3 = self.nesqd_to_norm * ne_pcm3**2
        flux_pkpc3 = self.fluxcalc.get(T_keV, Z_solar, norm_pkpc3)
        v['flux_cuml_%g_%g_ergpspcm2' % self.fluxrange] = N.cumsum(
            flux_pkpc3*self.out_vol_kpc3)

        flux_bolo_pkpc3 = self.fluxcalc_lumin_bolo.get(T_keV, Z_solar, norm_pkpc3)
        emiss_bolo = flux_bolo_pkpc3 * (self.flux_to_lumin/kpc3_cm3)
        v['L_bolo_ergpspcm3'] = emiss_bolo

        v['H_ergpcm3'] = (5/2) * v['ne_pcm3'] * (1 + 1/ne_nH) * v['T_keV'] * keV_erg
        v['tcool_yr'] = v['H_ergpcm3'] / v['L_bolo_ergpspcm3'] / yr_s


        # cumulative quantities
        # ---------------------
        # split quantities about shell midpoint, so result is
        # independent of binning
        fi, fo = fracMassHalf(N.arange(nshells), self.out_edges_kpc)
        def calc_cuml_midpt(vals):
            return vals*fi + N.concatenate(([0], N.cumsum(vals)[:-1]))

        Lshell = emiss_bolo * self.out_vol_cm3
        v['L_cuml_bolo_ergps'] = calc_cuml_midpt(Lshell)
        v['Mgas_cuml_Msun'] = calc_cuml_midpt(v['Mgas_Msun'])
        emiss_band = self.fluxcalc_lumin.get(T_keV, Z_solar, norm_pkpc3) * (
            self.flux_to_lumin / kpc3_cm3)
        Lshell_band = emiss_band * self.out_vol_cm3
        v['L_cuml_%g_%g_ergps' % self.luminrange] = calc_cuml_midpt(Lshell_band)

        # cumulative projected luminosity
        Lproj_bolo = self.out_projmatrix.dot(emiss_bolo)
        v['L_proj_cuml_bolo_ergps'] = calc_cuml_midpt(Lproj_bolo)
        Lproj_band = self.out_projmatrix.dot(emiss_band)
        v['L_proj_cuml_%g_%g_ergps' % self.luminrange] = calc_cuml_midpt(Lproj_band)

        # total mass (computed from g)
        v['Mtot_cuml_Msun'] = v['g_cmps2']*self.out_centre_cm**2/G_cgs/solar_mass_g
        # and the gas fraction (<r)
        v['fgas_cuml'] = v['Mgas_cuml_Msun'] / v['Mtot_cuml_Msun']
        # free fall fime
        rho_gpcm3 = (3/4./N.pi)*v['Mtot_cuml_Msun']*solar_mass_g / self.out_centre_cm**3
        v['tff_yr'] = N.sqrt(3*N.pi/32/G_cgs/rho_gpcm3) / yr_s
        v['tcool_tff'] = v['tcool_yr']/v['tff_yr']

        # Mdots
        Lshell_ergps = v['L_bolo_ergpspcm3'] * self.out_vol_cm3
        density_gpcm3 = v['ne_pcm3'] * mu_e * mu_g
        v['H_ergpg'] = v['H_ergpcm3'] / density_gpcm3
        v['Mdotpurecool_Msunpyr'] = (
            Lshell_ergps / v['H_ergpg'] / solar_mass_g * yr_s)
        v['Mdotpurecool_cuml_Msunpyr'] = N.cumsum(v['Mdotpurecool_Msunpyr'])

        # output mdot values go here
        v['Mdot_Msunpyr'] = N.zeros(nshells)
        v['Mdot_cuml_Msunpyr'] = N.zeros(nshells)

        # change in potential and enthalpy across each shell
        delta_pot_ergpg = N.concatenate((
                [0], v['potential_ergpg'][1:]-v['potential_ergpg'][:-1]))
        delta_H_ergpg = N.concatenate((
                [0], v['H_ergpg'][1:]-v['H_ergpg'][:-1]))

        Mdotcuml_gps = 0.
        for i in range(nshells):
            # total energy going into mdot in this shell, subtracting contribution
            # of matter which flows inwards
            E_tot_ergps = (
                Lshell_ergps[i] -
                Mdotcuml_gps*(delta_pot_ergpg[i] + delta_H_ergpg[i]))
            # energy comes from enthalpy plus change in potential
            E_tot_ergpg = v['H_ergpg'][i] + delta_pot_ergpg[i]

            Mdot_gps = E_tot_ergps / E_tot_ergpg
            v['Mdot_Msunpyr'][i] = Mdot_gps / solar_mass_g * yr_s
            Mdotcuml_gps += Mdot_gps
            v['Mdot_cuml_Msunpyr'][i] = Mdotcuml_gps / solar_mass_g * yr_s

        return v

    def loadChainFromFile(self, chainfname, burn=0, thin=10, randsamples=None):
        """Get list of parameter values from chain.

        :param chainfname: input chain HDF5 file
        :param burn: how many iterations to remove off input
        :param thin: discard every N entries
        :param randsamples: randomly sample N samples if set
        """

        return utils.loadChainFromFile(
            chainfname, self.pars,
            burn=burn, thin=thin, randsamples=randsamples,
        )

    def computePhysChains(self, chain):
        """Compute set of chains for each physical quantity.

        :param chain: list/array of free parameter values (can be loaded using loadChainFromFile)
        """

        uprint('Computing physical quantities from chain')
        pars = self.pars.copy()

        # iterate over input
        data = {}
        length = len(chain)
        for i, parvals in enumerate(chain):
            if i % 1000 == 0:
                uprint(' Step %i / %i (%.1f%%)' % (i, length, i*100/length))

            pars.setFree(parvals)

            ne_arr, T_arr, Z_arr = self.model.computeProfiles(pars, self.radii)
            g_arr, phi_arr = self.model.computeMassProfile(pars, self.radii)

            ne_resamp = self.rebinfn(ne_arr)
            T_resamp = self.rebinfn(T_arr)
            Z_resamp = self.rebinfn(Z_arr)
            g_resamp = self.rebinfn(g_arr)
            phi_resamp = self.rebinfn(phi_arr)

            physvals = self.calc(ne_resamp, T_resamp, Z_resamp, g_resamp, phi_resamp)
            for name, vals in physvals.items():
                if name not in data:
                    data[name] = []
                data[name].append(vals)

        # convert to numpy arrays
        for name in list(data):
            data[name] = N.array(data[name])

        return data

    def calcPhysChainStats(self, physchains, confint=68.269):
        """Take physical chains and compute profiles with 1 sigma ranges.

        :param physchains: physical chains computed from computePhysChains
        """

        uprint(' Computing medians')

        out = {}
        # convert to numpy arrays
        for name, chain in physchains.items():
            vals = N.array(chain)
            median, posrange, negrange = N.percentile(
                vals, [50, 50+confint/2, 50-confint/2], axis=0)

            prof = N.column_stack((median, posrange-median, negrange-median))
            out[name] = prof

        out['r_kpc'] = N.column_stack((
            0.5*(self.out_edges_kpc[1:]+self.out_edges_kpc[:-1]),
            +0.5*(self.out_edges_kpc[1:]-self.out_edges_kpc[:-1]),
            -0.5*(self.out_edges_kpc[1:]-self.out_edges_kpc[:-1])
        ))
        out['r_arcmin'] = out['r_kpc'] / (self.model.cosmo.as_kpc*60)

        return out

    def writeStatsToFile(self, fname, stats, mode='hdf5'):
        """Write computed statistics to output filename.

        Each physical variable is written with a column, with the 1-sigma
        +- error bars.

        :param mode: write file mode (only hdf5 supported so far)
        """


        if mode == 'hdf5':
            with h5py.File(fname, 'w') as f:
                for k in sorted(stats):
                    f[k] = stats[k]
                    f[k].attrs['vsz_twod_as_oned'] = 1

    def chainFileToStatsFile(
            self, chainfname, statfname, burn=0, thin=10, randsamples=None,
            confint=68.269,  mode='hdf5',
            ):
        """Simple wrapper to convert chain file to phys stats file.

        :param chainfname: input chain file name
        :param statfname: output statistics file name
        :param burn: how many iterations to remove off input
        :param thin: discard every N entries
        :param randsamples: select N random samples after burn (ignores thin)
        :param confint: confidence interval to output
        :param mode: write file mode (only hdf5 supported so far)
        """


        chain = utils.loadChainFromFile(
            chainfname, self.pars, burn=burn, thin=thin, randsamples=randsamples)
        physchain = self.computePhysChains(chain)
        stats = self.calcPhysChainStats(physchain, confint=confint)
        self.writeStatsToFile(statfname, stats)
