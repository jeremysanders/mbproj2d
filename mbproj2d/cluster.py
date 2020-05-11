# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
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

import numpy as N

from .model import SrcModelBase
from .ratecalc import RateCalc
from .profile import Radii
from .fast import addSBToImg

class ClusterBase(SrcModelBase):
    def __init__(
            self, name, pars, images, cosmo=None,
            NH_1022pcm2=0., cx=0., cy=0.):
        SrcModelBase.__init__(self, name, pars, images, cx=cx, cy=cy)

        self.cosmo = cosmo
        self.NH_1022pcm2 = NH_1022pcm2

    def computeProfiles(self, pars, radii):
        """Compute plasma profiles for use in physical profile computation

        :param pars: Pars object with parameters
        :param radii: Radii object with radii to compute for

        Returns (ne_prof, T_prof, Z_prof)
        """

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
            cx=cx, cy=cy,
            cosmo=cosmo,
            NH_1022pcm2=NH_1022pcm2,
        )

        self.ne_prof = ne_prof
        self.T_prof = T_prof
        self.Z_prof = Z_prof

        self.pixsize_Radii = {}  # radii indexed by pixsize
        self.image_RateCalc = {} # RateCalc for each image

        for img in images:
            pixsize_as = img.pixsize_as

            # Radii object is for a particular pixel size
            if pixsize_as not in self.pixsize_Radii:
                pixsize_kpc = images[0].pixsize_as * cosmo.as_kpc
                num = int(maxradius_kpc/pixsize_kpc)+1
                self.pixsize_Radii[pixsize_as] = Radii(pixsize_kpc, num)

            # make object to convert from plasma properties -> rates/kpc3
            self.image_RateCalc[img] = RateCalc(
                cosmo, img.rmf, img.arf, img.emin_keV, img.emax_keV,
                NH_1022pcm2)

    def compute(self, pars, imgarrs):
        """Add cluster model to images."""

        cy_as = pars['%s_cy' % self.name].v
        cx_as = pars['%s_cx' % self.name].v

        # get plasma profiles for each pixel size
        ne_arr = {}
        T_arr = {}
        Z_arr = {}
        for pixsize, radii in self.pixsize_Radii.items():
            ne_arr[pixsize] = self.ne_prof.compute(pars, radii)
            T_arr[pixsize] = self.T_prof.compute(pars, radii)
            Z_arr[pixsize] = self.Z_prof.compute(pars, radii)

        # add profiles to each image
        for img, imgarr in zip(self.images, imgarrs):
            pixsize_as = img.pixsize_as

            # calculate emissivity profile (per kpc3)
            emiss_arr_pkpc3 = self.image_RateCalc[img].get(
                ne_arr[pixsize_as], T_arr[pixsize_as], Z_arr[pixsize_as])

            # project emissivity to SB profile and convert to per pixel
            sb_arr = self.pixsize_Radii[pixsize_as].project(emiss_arr_pkpc3)
            sb_arr *= (self.cosmo.as_kpc*pixsize_as)**2

            # compute centre in pixels
            pix_cy = cy_as*img.invpixsize + img.origin[0]
            pix_cx = cx_as*img.invpixsize + img.origin[1]

            # add SB profile to image
            addSBToImg(1, sb_arr, pix_cx, pix_cy, imgarr)

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
