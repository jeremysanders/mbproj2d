# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
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

import numpy as N

from .model import ObjModelBase
from .ratecalc import RateCalc
from .profile import Radii
from .fast import addSBToImg

class ClusterNonHydro(ObjModelBase):

    def __init__(
            self, name, pars, images,
            cosmo=None,
            NH_1022pcm2=0.,
            ne_prof=None, T_prof=None, Z_prof=None,
            expmap=None,
            maxradius_kpc=3000.,
            cx=0., cy=0.
    ):
        ObjModelBase.__init__(
            self, name, pars, images, expmap=expmap, cx=cx, cy=cy)

        self.cosmo = cosmo
        self.NH_1022pcm2 = NH_1022pcm2
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

    def compute(self, imgarrs):
        """Add cluster model to images."""

        cy_as = self.pars['%s_cy' % self.name].vout()
        cx_as = self.pars['%s_cx' % self.name].vout()

        # get plasma profiles for each pixel size
        ne_arr = {}
        T_arr = {}
        Z_arr = {}
        for pixsize, radii in self.pixsize_Radii.items():
            ne_arr[pixsize] = self.ne_prof.compute(radii)
            T_arr[pixsize] = self.T_prof.compute(radii)
            Z_arr[pixsize] = self.Z_prof.compute(radii)

        # add profiles to each image
        for img, imgarr in zip(self.images, imgarrs):
            pixsize_as = img.pixsize_as

            # calculate emissivity profile (per kpc3)
            emiss_arr_pkpc3 = self.image_RateCalc[img].getRate(
                T_arr[pixsize_as], Z_arr[pixsize_as], ne_arr[pixsize_as])

            # project emissivity to SB profile and convert to per pixel
            sb_arr = self.pixsize_Radii[pixsize_as].project(emiss_arr_pkpc3)
            sb_arr *= (self.cosmo.as_kpc*pixsize_as)**2

            if self.expmap is None:
                # inefficient
                expmap = N.ones(img.shape, dtype=N.float32)
            else:
                expmap = img.expmaps[self.expmap]

            # compute centre in pixels
            pcy = cy_as*img.invpixsize + img.origin[0]
            pcx = cx_as*img.invpixsize + img.origin[1]

            # add SB profile to image
            addSBToImg(1, sb_arr, pcx, pcy, expmap, imgarr)
