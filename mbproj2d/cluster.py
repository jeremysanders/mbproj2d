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

from .model import ObjModelBase
from .profile import Radii

def compute_emissivity(image, radii, NH, cosmo, ne_arr, T_arr, Z_arr):
    pass

def project_emissivity(imgarr, radii, emissivity):
    pass

class ClusterNonHydro(ObjModelBase):

    def __init__(
            self, name, pars, images,
            cosmo=None,
            NH_1022cm2=0.,
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

        self.pixsize_radii = {}  # radii indexed by pixsize
        for img in images:
            pixsize_as = img.pixsize_as
            if pixsize_as not in self.pixsize_radii:
                pixsize_kpc = images[0].pixsize_as * cosmo.as_kpc
                edges_kpc = N.arange(int(maxradius_kpc/pixsize_kpc)+1)*pixsize_kpc
                self.pixsize_radii[pixsize_as] = Radii(edges_kpc)

    def compute(self, imgarrs):

        # get intrinsic profiles for each pixel size
        ne_arr = {}
        T_arr = {}
        Z_arr = {}
        for pixsize, radii in self.pixsize_radii.items():
            ne_arr[pixsize] = self.ne_prof.compute(radii)
            T_arr[pixsize] = self.T_prof.compute(radii)
            Z_arr[pixsize] = self.Z_prof.compute(radii)

        for image in self.images:
            pixsize = image.pixsize_as
            emiss_arr = compute_emissivity(
                image, self.pixsize_radii[pixsize],
                self.NH_1022pcm2, self.cosmo,
                ne_arr[pixsize], T_arr[pixsize], Z_arr[pixsize]
            )
