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

class Image:
    def __init__(
            self, imgid, imagearr,
            emin_keV=0.5, emax_keV=2.0,
            rmf='image.rmf',
            arf='image.arf',
            pixsize_as=1.0,
            expmaps=None,
            mask=None,
            psf=None,
            origin=(0,0),
    ):
        """Image class holds all information about an image.

        :param imgid: unique id for image (str or int)
        :param imagearr: numpy image array for image
        :param float emin_keV: minimum energy
        :param float emax_keV: maximum energy
        :param rmf: response matrix file
        :param arf: ancillary response matrix file
        :param pixsize_as: size of pixels in arcsec
        :param expmaps: list or dict of numpy exposure map arrays (different components can use different exposure maps, if needed)
        :param mask: numpy mask array (None means no mask)
        :param psf: PSF object
        :param origin: position (y,x) coordinates are measured relative to (should be same position in all images)
        """

        self.imgid = imgid
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.rmf = rmf
        self.arf = arf
        self.imagearr = imagearr.astype(N.float32)
        self.shape = imagearr.shape
        self.pixsize_as = pixsize_as
        self.invpixsize = 1/pixsize_as

        # mask should be -1 (included) or 0 (excluded), for use in simd
        if mask is None:
            self.mask = N.empty(self.shape, dtype=N.int32)
            self.mask.fill(-1)
        else:
            self.mask = N.where(mask != 0, -1, 0).astype(N.int32)

        self.expmaps = expmaps
        self.psf = psf
        self.origin = origin
