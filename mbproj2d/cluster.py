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
import numpy as N

from .model import SrcModelBase
from .ratecalc import ApecRateCalc
from .profile import Radii
from .fast import addSBToImg_Comb
from .par import Par
from . import utils
from .physconstants import kpc_cm, Mpc_cm, kpc3_cm3, mu_e, mu_g, G_cgs, P_keV_to_erg

class ClusterBase(SrcModelBase):
    """Base cluster model class.

    This does quite a lot of work for the child classes, as it takes
    the computed ne,T,Z profiles and makes the images.
    """

    def __init__(
            self, name, pars, images, cosmo=None,
            NH_1022pcm2=0.,
            maxradius_kpc=3000.,
            cx=0., cy=0.):
        SrcModelBase.__init__(self, name, pars, images, cx=cx, cy=cy)

        self.cosmo = cosmo
        self.NH_1022pcm2 = NH_1022pcm2
        self.maxradius_kpc = maxradius_kpc

        self.pixsize_Radii = {}  # radii indexed by pixsize
        self.image_RateCalc = {} # RateCalc for each image

        # factor to convert from ne**2 -> norm/kpc3
        self.nesqd_to_norm = utils.calcNeSqdToNormPerKpc3(cosmo)

        for img in images:
            pixsize_as = img.pixsize_as

            # Radii object is for a particular pixel size
            if pixsize_as not in self.pixsize_Radii:
                pixsize_kpc = pixsize_as * cosmo.as_kpc
                num = int(maxradius_kpc/pixsize_kpc)+1
                self.pixsize_Radii[pixsize_as] = Radii(pixsize_kpc, num)

            # make object to convert from plasma properties to rates
            self.image_RateCalc[img] = ApecRateCalc(
                img.rmf, img.arf, img.emin_keV, img.emax_keV,
                NH_1022pcm2, cosmo.z)

    def compute(self, pars, imgarrs):
        """Add cluster model to images."""

        cy_as = pars[f'{self.name}_cy'].v
        cx_as = pars[f'{self.name}_cx'].v

        # Optional parameters:
        #  ellipticity (0..1)
        name = f'{self.name}_e'
        e = 1 if name not in pars else pars[name].v
        #  slosh amplitude (0..1)
        name = f'{self.name}_slosh'
        slosh = 0 if name not in pars else pars[name].v
        #  angle of slosh, ellipticity or multipole (radians)
        name = f'{self.name}_theta'
        theta = 0 if name not in pars else pars[name].v
        #  multipole index (1,2,3,4), or 0 to disable
        name = f'{self.name}_mulm'
        mulm = 0 if name not in pars else int(pars[name].v)
        #  multipole magnitude (0..1)
        name = f'{self.name}_mulmag'
        mulmag = 0 if name not in pars else pars[name].v

        # get plasma profiles for each pixel size
        norm_arr = {}
        T_arr = {}
        Z_arr = {}
        for pixsize, radii in self.pixsize_Radii.items():
            neprof, Tprof, Zprof = self.computeProfiles(pars, radii)
            norm_arr[pixsize] = self.nesqd_to_norm * neprof**2
            T_arr[pixsize] = Tprof
            Z_arr[pixsize] = Zprof

        # add profiles to each image
        for img, imgarr in zip(self.images, imgarrs):
            pixsize_as = img.pixsize_as

            # calculate emissivity profile (per kpc3)
            emiss_arr_pkpc3 = self.image_RateCalc[img].get(
                T_arr[pixsize_as], Z_arr[pixsize_as], norm_arr[pixsize_as])

            # project emissivity to SB profile and convert to per pixel
            sb_arr = self.pixsize_Radii[pixsize_as].project(emiss_arr_pkpc3)
            sb_arr *= (self.cosmo.as_kpc*pixsize_as)**2

            # compute centre in pixels
            pix_cy = cy_as*img.invpixsize + img.origin[0]
            pix_cx = cx_as*img.invpixsize + img.origin[1]

            # add SB profile to image
            addSBToImg_Comb(
                1, sb_arr, pix_cx, pix_cy, e, slosh, theta, mulm, mulmag, imgarr)

    def computeProfiles(self, pars, radii):
        """Compute plasma profiles for use in physical profile computation

        :param pars: Pars object with parameters
        :param radii: Radii object with radii to compute for

        Returns (ne_prof, T_prof, Z_prof)
        """

    def computeMassProfile(self, pars, radii):
        """Return g and Phi profiles for cluster, if available."""
        empty = N.zeros(radii.num)
        return empty, empty

class ClusterNonHydro(ClusterBase):
    """Model for a cluster, given density, temperature and metallicity profiles."""

    def __init__(
            self, name, pars, images,
            cosmo=None,
            NH_1022pcm2=0.,
            ne_prof=None, T_prof=None, Z_prof=None,
            maxradius_kpc=3000.,
            cx=0., cy=0.
    ):
        """
        :param name: name of model to apply to parameters
        :param pars: Pars() object to define parameters
        :param images: list of Image objects
        :param cosmo: Cosmology object
        :param NH_1022pcm2: Column density
        :param ne_prof: Profile object for density
        :param T_prof: Profile object for temperature
        :param Z_prof: Profile object for metallicity
        :param maxradius_kpc: Compute profile out to this radius
        :param cx: cluster centre (arcsec)
        :param cy: cluster centre (arcsec)
        """

        ClusterBase.__init__(
            self, name, pars, images,
            cosmo=cosmo,
            NH_1022pcm2=NH_1022pcm2,
            maxradius_kpc=maxradius_kpc,
            cx=cx, cy=cy,
        )

        self.ne_prof = ne_prof
        self.T_prof = T_prof
        self.Z_prof = Z_prof

    def prior(self, pars):
        return (
            self.ne_prof.prior(pars) +
            self.T_prof.prior(pars) +
            self.Z_prof.prior(pars)
        )

    def computeProfiles(self, pars, radii):
        """Compute plasma profiles.

        :param pars: Pars object with parameters
        :param radii: Radii object with radii to compute for

        Returns (ne_prof, T_prof, Z_prof)
        """

        ne_arr = self.ne_prof.compute(pars, radii)
        T_arr = self.T_prof.compute(pars, radii)
        Z_arr = self.Z_prof.compute(pars, radii)

        return ne_arr, T_arr, Z_arr

def computeGasAccn(radii, ne_prof):
    """Compute acceleration due to gas mass for density profile
    given."""

    # mass in each shell
    masses_g = ne_prof * radii.vol_kpc3 * ( kpc3_cm3 * mu_e * mu_g)

    # cumulative mass interior to each shell
    Minterior_g = N.cumsum( N.hstack( ([0.], masses_g[:-1]) ) )

    # this is the mean acceleration on the shell, computed as total
    # force from interior mass divided by the total mass:
    #   ( Int_{r=R1}^{R2} (G/r**2) *
    #                     (M + Int_{R=R1}^{R} 4*pi*R^2*rho*dR) *
    #                     4*pi*r^2*rho*dR ) / (
    #   (4./3.*pi*(R2**3-R1**3)*rho)
    rout, rin = radii.outer_kpc*kpc_cm, radii.inner_kpc*kpc_cm
    gmean = G_cgs*(
        3*Minterior_g +
        ne_prof*(mu_e*mu_g*math.pi)*(
            (rout-rin)*((rout+rin)**2 + 2*rin**2)))  / (
        rin**2 + rin*rout + rout**2 )

    return gmean

class ClusterHydro(ClusterBase):
    """Hydrostatic model for cluster, given density, mass model and metallicity profile."""

    # we clip to range otherwise the fitting fails
    Tmin = 0.06
    Tmax = 60.

    def __init__(
            self, name, pars, images,
            cosmo=None,
            NH_1022pcm2=0.,
            ne_prof=None, mass_prof=None, Z_prof=None,
            gas_has_mass=True,
            maxradius_kpc=3000.,
            cx=0., cy=0.
    ):
        """
        :param name: name of model to apply to parameters
        :param pars: Pars() object to define parameters
        :param images: list of Image objects
        :param cosmo: Cosmology object
        :param NH_1022pcm2: Column density
        :param ne_prof: Profile object for density
        :param mass_prof: ProfileMass object for mass profile
        :param Z_prof: Profile object for metallicity
        :param gas_has_mass: Add gas mass to calculations
        :param maxradius_kpc: Compute profile out to this radius
        :param cx: cluster centre (arcsec)
        :param cy: cluster centre (arcsec)
        """

        ClusterBase.__init__(
            self, name, pars, images,
            cosmo=cosmo,
            NH_1022pcm2=NH_1022pcm2,
            maxradius_kpc=maxradius_kpc,
            cx=cx, cy=cy,
        )

        pars['%s_Pout_logergpcm3' % name] = Par(-32., minval=-37, maxval=-18)
        self.ne_prof = ne_prof
        self.mass_prof = mass_prof
        self.Z_prof = Z_prof
        self.gas_has_mass = gas_has_mass

    def prior(self, pars):
        return (
            self.ne_prof.prior(pars) +
            self.mass_prof.prior(pars) +
            self.Z_prof.prior(pars)
        )

    def computeProfiles(self, pars, radii):
        """Compute plasma profiles assuming hydrostatic equilibrium

        :param pars: Pars object with parameters
        :param radii: Radii object with radii to compute for

        Returns (ne_prof, T_prof, Z_prof)
        """

        P0_ergpcm3 = math.exp(pars['%s_Pout_logergpcm3' % self.name].v)

        Z_solar = self.Z_prof.compute(pars, radii)
        ne_pcm3 = self.ne_prof.compute(pars, radii)

        g_cmps2, Phi_arr = self.mass_prof.compute(pars, radii)

        # prevent formulae blowing up
        ne_pcm3 = N.clip(ne_pcm3, 1e-99, 1e99)

        # optionally include effect of gas mass on accn
        if self.gas_has_mass:
            g_cmps2 += computeGasAccn(radii, ne_pcm3)

        # changes in pressure in outer and inner halves of bin (around centre)
        # this is a bit more complex than needed, but we can change the midpt
        P_pcm = g_cmps2 * ne_pcm3 * (mu_e*mu_g)
        mid_cm = radii.cent_cm
        deltaP_out = (radii.outer_cm - mid_cm) * P_pcm
        deltaP_in = (mid_cm - radii.inner_cm) * P_pcm

        # combine halves and include outer pressure to get incremental deltaP
        deltaP_halves = N.ravel( N.column_stack((deltaP_in, deltaP_out)) )
        deltaP_ergpcm3 = N.concatenate((deltaP_halves[1:], [P0_ergpcm3]))

        # add up contributions inwards to get total pressure,
        # discarding pressure between shells
        P_ergpcm3 = N.cumsum(deltaP_ergpcm3[::-1])[::-2]

        # calculate temperatures given pressures and densities
        T_keV = P_ergpcm3 / (P_keV_to_erg * ne_pcm3)

        T_keV = N.clip(T_keV, self.Tmin, self.Tmax)

        return ne_pcm3, T_keV, Z_solar

    def computeMassProfile(self, pars, radii):
        """Compute g and potential given parameters."""

        g_prof, pot_prof = self.mass_prof.compute(pars, radii)

        if self.gas_has_mass:
            ne_prof = self.ne_prof.compute(pars, radii)
            g_prof += computeGasAccn(radii, ne_prof)

        return g_prof, pot_prof

class EmissionMeasureCluster(ClusterBase):
    """Less physical cluster model, where we parameterize using a projected emission measure profile.

    Here T_prof and Z_prof are also projected quantities
    emiss_prof has units of cm^-5
    """

    def __init__(self, name, pars, images,
                 cosmo=None,
                 NH_1022pcm2=0.,
                 emiss_prof=None, T_prof=None, Z_prof=None,
                 maxradius_kpc=3000.,
                 cx=0., cy=0.):

        ClusterBase.__init__(
            self, name, pars, images,
            cosmo=cosmo,
            NH_1022pcm2=NH_1022pcm2,
            maxradius_kpc=maxradius_kpc,
            cx=cx, cy=cy)

        self.emiss_prof = emiss_prof
        self.T_prof = T_prof
        self.Z_prof = Z_prof

    def prior(self, pars):
        return (
            self.emiss_prof.prior(pars) +
            self.T_prof.prior(pars) +
            self.Z_prof.prior(pars)
        )

    def compute(self, pars, imgarrs):
        """Compute cluster images.

        This completely overrides the base class."""

        cy_as = pars[f'{self.name}_cy'].v
        cx_as = pars[f'{self.name}_cx'].v

        # Optional parameters:
        #  ellipticity (0..1)
        name = f'{self.name}_e'
        e = 1 if name not in pars else pars[name].v
        #  slosh amplitude (0..1)
        name = f'{self.name}_slosh'
        slosh = 0 if name not in pars else pars[name].v
        #  angle of slosh, ellipticity or multipole (radians)
        name = f'{self.name}_theta'
        theta = 0 if name not in pars else pars[name].v
        #  multipole index (1,2,3,4), or 0 to disable
        name = f'{self.name}_mulm'
        mulm = 0 if name not in pars else int(pars[name].v)
        #  multipole magnitude (0..1)
        name = f'{self.name}_mulmag'
        mulmag = 0 if name not in pars else pars[name].v

        # get profiles for each pixel size
        emiss_arr = {}
        T_arr = {}
        Z_arr = {}
        for pixsize, radii in self.pixsize_Radii.items():
            emiss_arr[pixsize] = self.emiss_prof.compute(pars, radii)
            T_arr[pixsize] = self.T_prof.compute(pars, radii)
            Z_arr[pixsize] = self.Z_prof.compute(pars, radii)

        # add profiles to each image
        for img, imgarr in zip(self.images, imgarrs):
            pixsize_as = img.pixsize_as
            pixsize_cm = pixsize_as * self.cosmo.as_kpc * kpc_cm

            em_scale = 1e-14 * pixsize_cm**2 / (
                4*N.pi * (self.cosmo.D_A*Mpc_cm*(1+self.cosmo.z))**2)
            norm_ppix2 = emiss_arr[pixsize_as]*em_scale

            sb_arr = self.image_RateCalc[img].get(
                T_arr[pixsize_as], Z_arr[pixsize_as], norm_ppix2)
            sb_arr = sb_arr.astype(N.float32)

            # compute centre in pixels
            pix_cy = cy_as*img.invpixsize + img.origin[0]
            pix_cx = cx_as*img.invpixsize + img.origin[1]

            # add SB profile to image
            addSBToImg_Comb(
                1, sb_arr, pix_cx, pix_cy, e, slosh, theta, mulm, mulmag, imgarr)
