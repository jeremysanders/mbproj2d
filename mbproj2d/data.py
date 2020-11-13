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

import numpy as N
from scipy.special import gammaln

from . import utils
from . import fast
from astropy.wcs import WCS

class Image:
    """Image class details images to fit.

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
    :param origin: position in pixels (y,x) coordinates are measured relative to (should be same position in all images)
    :param wcs: optional WCS stored with this image
    :param optimal_size: expand images to be optimal size for PSF convolution
    """

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
            wcs=None,
            optimal_size=True,
    ):
        self.img_id = img_id
        self.emin_keV = emin_keV
        self.emax_keV = emax_keV
        self.rmf = rmf
        self.arf = arf
        self.pixsize_as = pixsize_as
        self.invpixsize = 1/pixsize_as
        self.wcs = wcs

        if optimal_size:
            imagearr, expmaps, mask = self._expandOptimal(
                imagearr, expmaps, mask)

        # copy image
        self.shape = imagearr.shape
        self.imagearr = utils.empty_aligned(self.shape, dtype=N.float32)
        self.imagearr[:,:] = imagearr

        if self.shape[0] % 2 !=0 or self.shape[1] % 2 != 0:
            raise RuntimeError('Input images must have even numbers of pixels')

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

    def _expandOptimal(self, imagearr, expmaps, mask):
        """Expand image sizes to be optimal for FFT speed

        pyfftw works fastest on images which are factors of
        2**a 3**b 5**c 7**d 11**e 13**f, where e+f is either 0 or 1, and a to d >=0

        """

        # get list of fast sizes (reliable up to 2**16)
        primes = [2,3,5,7]
        fastdims = set(primes)
        for nprime in range(16):
            for p in primes:
                for v in list(fastdims):
                    if p*v <= 65536:
                        fastdims.add(p*v)
        # also factors of 11 and 13 of above are fast
        for f in 11, 13:
            for v in list(fastdims):
                if v*f <= 65536:
                    fastdims.add(v*f)
        # get rid of odd factors
        for d in list(fastdims):
            if d%2 != 0:
                fastdims.remove(d)
        fastdims = N.array(sorted(fastdims))

        # find next largest size
        odim0, odim1 = imagearr.shape
        dim0 = fastdims[N.searchsorted(fastdims, odim0)]
        dim1 = fastdims[N.searchsorted(fastdims, odim1)]

        # no expansion necessary
        if odim0==dim0 and odim1==dim1:
            return imagearr, expmaps, mask

        # expand image with 0 at edge
        newimage = N.zeros((dim0, dim1), dtype=N.float32)
        newimage[:odim0,:odim1] = imagearr

        # expand exposure maps with 0 at edge
        if expmaps is None:
            newexpmaps = None
        else:
            newexpmaps = {}
            for name, expmap in expmaps.items():
                newexpmap = N.zeros((dim0, dim1), dtype=N.float32)
                newexpmap[:odim0,:odim1] = expmap
                newexpmaps[name] = newexpmap

        # expand mask with 0 at edge
        if mask is None:
            mask = N.ones((odim0, odim1), dtype=N.int32)
        newmask = N.zeros((dim0, dim1), dtype=N.int32)
        newmask[:odim0,:odim1] = N.where(mask, 1, 0)

        return newimage, newexpmaps, newmask

    def binUp(self, factor):
        """Return binned copy of image.

        :param factor: integer bin factor
        """

        neworigin = (self.origin[0]/factor, self.origin[1]/factor)

        # bin up cts
        newimg = utils.binImage(self.imagearr, factor)

        # mask, keeping pixels where there are no masked pixels in any subpixel
        newmask = (
            utils.binImage(N.where(self.mask, 1, 0), factor) == 
            utils.binImage(N.ones(self.mask.shape, dtype=N.int32), factor)
        )

        # compute binned up mean exposure maps
        newexpmaps = {
            name: utils.binImage(expmap, factor, mean=True)
            for name, expmap in self.expmaps.items()
        }

        # rescale WCS if given
        newwcs = None
        if self.wcs is not None:
            hdr = self.wcs.to_header()
            hdr['CDELT1'] = hdr['CDELT1'] * factor
            hdr['CDELT2'] = hdr['CDELT2'] * factor
            hdr['CRPIX1'] = (hdr['CRPIX1']-1)/factor + 1
            hdr['CRPIX2'] = (hdr['CRPIX2']-1)/factor + 1
            newwcs = WCS(hdr)

        return Image(
            self.img_id,
            newimg,
            emin_keV=self.emin_keV, emax_keV=self.emax_keV,
            rmf=self.rmf, arf=self.arf,
            pixsize_as=self.pixsize_as*factor,
            expmaps=newexpmaps,
            mask=newmask,
            psf=self.psf,
            origin=neworigin,
            wcs=newwcs,
        )

class PSF:
    """PSF modelling class.

    :param img: 2D PSF image
    :param pixsize_as: size of pixels in arcsec
    :param origin: (y, x) origin of PSF centre
    """

    def __init__(self, img, pixsize_as=1.0, origin=None):
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
