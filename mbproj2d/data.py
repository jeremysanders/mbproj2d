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
from scipy.special import gammaln

from . import utils
from . import fast

class Image:
    """Image class details images to fit."""

    def __init__(
            self, img_id, imagearr,
            emin_keV=0.5, emax_keV=2.0,
            rmf='image.rmf',
            arf='image.arf',
            pixsize_as=1.0,
            expmaps=None,
            mask=None,
            psf=None,
            origin=(0,0),
    ):
        """
        :param img_id: unique id for image (str or int)
        :param imagearr: numpy image array for image
        :param float emin_keV: minimum energy
        :param float emax_keV: maximum energy
        :param rmf: response matrix file
        :param arf: ancillary response matrix file
        :param pixsize_as: size of pixels in arcsec
        :param expmaps: dict of numpy exposure map arrays (different components can use different exposure maps, if needed)
        :param mask: numpy mask array (None means no mask)
        :param psf: PSF object
        :param origin: position (y,x) coordinates are measured relative to (should be same position in all images)
        """

        self.img_id = img_id
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.rmf = rmf
        self.arf = arf
        self.shape = imagearr.shape
        self.pixsize_as = pixsize_as
        self.invpixsize = 1/pixsize_as

        if self.shape[0] % 2 !=0 or self.shape[1] % 2 != 0:
            raise RuntimeError('Input images must have even numbers of pixels')

        # copy image
        self.imagearr = utils.empty_aligned(self.shape, dtype=N.float32)
        self.imagearr[:,:] = imagearr

        # mask should be -1 (included) or 0 (excluded), for use in simd
        self.mask = utils.zeros_aligned(self.shape, dtype=N.int32)
        if mask is None:
            self.mask.fill(-1)
        else:
            self.mask[:,:] = N.where(mask, -1, 0)

        self.expmaps = expmaps

        if psf is None:
            self.psf = None
        else:
            # copy so image-specific parts are kept separate
            self.psf = psf.copy()
            self.psf.matchImage(self.shape, pixsize_as)

        self.origin = origin

    def expandArrays(self, shape):
        """Expand arrays to match shape given.

        (not tested)
        """
        temp = utils.zeros_aligned(shape, dtype=N.float32)
        temp[:self.shape[0],:self.shape[1]] = self.imagearr
        self.imagearr = temp

        temp = utils.zeros_aligned(shape, dtype=N.int32)
        temp[:self.shape[0],:self.shape[1]] = self.mask
        self.mask = temp

        if self.expmaps is not None:
            for key in self.expmaps:
                temp = utils.zeros_aligned(shape, dtype=N.float32)
                temp[:self.shape[0],:self.shape[1]] = self.expmaps[key]
                self.expmaps[key] = temp

        self.shape = shape

class PSF:
    """PSF modelling class."""

    def __init__(self, img, pixsize_as=1.0, origin=None):
        """
        :param img: 2D PSF image
        :param pixsize_as: size of pixels in arcsec
        :param origin: (y, x) origin of PSF centre
        """
        self.img = img
        if origin is None:
            self.origin = img.shape[0]/2, img.shape[1]/2
        else:
            self.origin = origin
        self.pixsize_as = pixsize_as

        self.resample = None
        self.fft = None

    def copy(self):
        return PSF(
            N.array(self.img),
            pixsize_as=self.pixsize_as,
            origin=self.origin,
        )

    def matchImage(self, imgshape, img_pixsize_as):
        """Adjust PSF to different pixel size and image shape."""

        self.resample = utils.empty_aligned(imgshape, dtype=N.float32)
        fast.resamplePSFImage(
            self.img.astype(N.float32),
            self.resample,
            psf_pixsize=self.pixsize_as,
            img_pixsize=img_pixsize_as,
            psf_ox=self.origin[1],
            psf_oy=self.origin[0],
        )
        self.fft = utils.run_rfft2(self.resample)

    def applyTo(self, inimg, minval=1e-10):
        """Convolve image with PSF."""

        imgfft = utils.run_rfft2(inimg)
        imgfft *= self.fft
        convolved = utils.run_irfft2(imgfft)
        # make sure convolution is positive
        fast.clip2DMax(convolved, minval)
        inimg[:,:] = convolved
