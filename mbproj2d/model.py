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

from astropy.io import fits

from .par import Par, PriorGaussian, PriorBoundedGaussian
from . import utils
from . import ratecalc

class TotalModel:
    """Combined model for data.

    :param pars: Pars object (currently unused here)
    :param images: list of Image objects
    :param src_models: list of source models
    :param src_expmap: name of exposure map to use for sources
    :param back_models: list of background models
    """

    def __init__(
            self, pars, images, src_models=None, src_expmap=None,
            back_models=None):

        self.images = images
        self.src_models = src_models
        self.src_expmap = src_expmap
        self.back_models = back_models

    def compute(
            self,
            pars,
            apply_psf=True, apply_expmap=True,
            apply_src=True, apply_back=True,
    ):
        """Return a list of image arrays for the models.
        :param pars: Pars() object with parameters
        :param apply_psf: whether to convolve with PSF
        :param apply_expmap: whether to multiply by exposure map
        :param apply_src: apply source models
        :param apply_back: apply background models
        """

        # output images for each Image
        imgarrs = [
            utils.zeros_aligned(image.shape, dtype=N.float32)
            for image in self.images
        ]

        # add the models to the image
        if self.src_models and apply_src:
            # actual model computation
            for model in self.src_models:
                model.compute(pars, imgarrs)

            # optional scaling of model
            for imgarr, image in zip(imgarrs, self.images):
                scale_name = f'src_logscale_{image.img_id}'
                if scale_name in pars:
                    imgarr *= math.exp(pars[scale_name].v)

            # convolve with PSF
            if apply_psf:
                for imgarr, image in zip(imgarrs, self.images):
                    if image.psf is not None:
                        image.psf.applyTo(imgarr)

            # apply exposure map
            if apply_expmap and self.src_expmap is not None:
                for imgarr, image in zip(imgarrs, self.images):
                    imgarr *= image.expmaps[self.src_expmap]

        # add on background
        if self.back_models and apply_back:
            for model in self.back_models:
                model.compute(pars, imgarrs)

        return imgarrs

    def compute_separate(
            self,
            pars,
            apply_psf=True, apply_expmap=True,
    ):
        """Compute model for each component separately and combined.
        :param pars: Pars() object with parameters
        :param apply_psf: whether to convolve with PSF
        :param apply_expmap: whether to multiply by exposure map
        """

        out = {}

        def make_blank_images():
            return [
                utils.zeros_aligned(image.shape, dtype=N.float32)
                for image in self.images
            ]

        # source models
        for model in self.src_models:
            imgarrs = make_blank_images()
            model.compute(pars, imgarrs)

            # convolve with PSF
            if apply_psf:
                for imgarr, image in zip(imgarrs, self.images):
                    if image.psf is not None:
                        image.psf.applyTo(imgarr)

            # apply exposure map
            if apply_expmap and self.src_expmap is not None:
                for imgarr, image in zip(imgarrs, self.images):
                    imgarr *= image.expmaps[self.src_expmap]

            out[model.name] = imgarrs

        # background components
        for model in self.back_models:
            imgarrs = make_blank_images()
            model.compute(pars, imgarrs)
            out[model.name] = imgarrs

        # add up total
        totimgarrs = make_blank_images()
        for name in out:
            for i, arr in enumerate(out[name]):
                totimgarrs[i] += arr
        out['total'] = totimgarrs

        return out

    def prior(self, pars):
        """Given parameters, compute prior.

        This includes the contribution from parameters, the object
        models and background models.
        """

        total = pars.calcPrior()
        if self.src_models:
            for model in self.src_models:
                total += model.prior(pars)
        if self.back_models:
            for model in self.back_models:
                total += model.prior(pars)
        return total

    def writeToFits(self, filename, pars, mask=True, apply_psf=True,
                    apply_expmap=True, apply_src=True, apply_back=True,
                    trim_original=False):
        """Write model as a series of HDUs in a FITS file.

        :param filename: FITS filename
        :param pars: pars dictionary
        :param mask: mask out masked pixels
        :param apply_psf: convolve with PSF
        :param apply_expmap: multiply by exposure map
        :param apply_src: include source models
        :param apply_back: include background models
        :param trim_original: write to original size
        """

        hdus = [fits.PrimaryHDU()]
        modimgs = self.compute(
            pars,
            apply_psf=apply_psf, apply_expmap=apply_expmap,
            apply_src=apply_src, apply_back=apply_back)
        for image, mod in zip(self.images, modimgs):
            if mask:
                mod[image.mask==0] = N.nan
            if trim_original:
                mod = mod[:image.orig_shape[0],:image.orig_shape[1]]

            hdu = fits.ImageHDU(mod)
            hdu.header['EXTNAME'] = image.img_id
            if image.wcs is not None:
                hdu.header.extend(image.wcs.to_header().cards)
            hdus.append(hdu)
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=True)

class BackModelBase:
    """Base background model. Does nothing.

    :param name: name of model
    :param pars: Pars() object
    :param images: list of Image() objects
    :param expmap: exposure map name to lookup in Image
    """

    def __init__(self, name, pars, images, expmap=None):
        self.name = name
        self.images = images
        self.expmap = expmap

    def compute(self, pars, imgarrs):
        pass

    def prior(self, pars):
        return 0

class BackModelFlat(BackModelBase):
    """Flat background model.

    :param name: name of model
    :param pars: dict of parameters
    :param images: list of data.Image objects
    :param bool log: apply log scaling to value of background
    :param bool normarea: normalise background to per sq arcsec
    :param defval: default parameter
    :param expmap: name or index of exposure map to use (if any)
    """

    def __init__(
            self, name, pars, images,
            log=False, normarea=False,
            defval=0.,
            expmap=None
    ):
        """A flat background model.
        """
        BackModelBase.__init__(self, name, pars, images, expmap=expmap)
        for image in images:
            if log:
                pars['%s_%s' % (name, image.img_id)] = Par(defval)
            else:
                pars['%s_%s' % (name, image.img_id)] = Par(defval, minval=0.)
        pars['%s_scale' % name] = Par(
            1.0, prior=PriorGaussian(1.0, 0.05), frozen=True)
        self.normarea = normarea
        self.log = log
        self.expmap = expmap

    def compute(self, pars, imgarrs):
        scale = pars['%s_scale' % self.name].v
        for image, imgarr in zip(self.images, imgarrs):
            v = pars['%s_%s' % (self.name, image.img_id)].v
            if self.log:
                v = math.exp(min(v, 100))
            if self.normarea:
                v *= image.pixsize_as**2
            v *= scale
            if self.expmap is not None:
                v *= image.expmaps[self.expmap]
            imgarr += v

class BackModelVigNoVig(BackModelBase):
    """A background model with vignetted and non-vignetted components.

    The X_vf holds the vignetted fraction. This is an unbounded
    parameter which is transformed to the vignetting fraction via the
    sigmoid function. By default it has a Gaussian prior about 0.

    All parameters are log units per square arcsec

    Note that the unvignetted exposure map is multiplied by median of
    the ratio of the vignetted to unvignetted if rescale_novig is
    True. This helps to make a seamless transition between the two as
    a function of X_vf. This reduces the degeneracies between the X_vf
    and normalisation parameters.

    :param name: name of model
    :param pars: dict of parameters
    :param images: list of data.Image objects
    :param defval: default parameter
    :param expmap: name or index of vignetted exposure map to use
    :param expmap_novig: name or index of unvignetted exposure map to use
    :param rescale_vig: rescale non-vignetted map by ratio of vig to non-vig
    """

    def __init__(
            self, name, pars, images,
            defval=0.,
            expmap='expmap', expmap_novig='expmap_novig',
            rescale_novig=True
    ):
        BackModelBase.__init__(self, name, pars, images, expmap=expmap)
        pars['%s_logscale' % name] = Par(
            0, prior=PriorGaussian(0, 0.05), frozen=True)

        self.vigratios = []
        for image in images:
            imgkey = '%s_%s' % (name, image.img_id)
            pars[imgkey] = Par(defval, minval=-30, maxval=30)
            pars['%s_vf' % imgkey] = Par(0.1, prior=PriorBoundedGaussian(
                0, 1, minval=-5, maxval=5))
            if rescale_novig:
                self.vigratios.append( N.median(
                    (image.expmaps[expmap] /
                     image.expmaps[expmap_novig])[image.mask != 0]
                ))
            else:
                self.vigratios.append(1)

        self.expmap = expmap
        self.expmap_novig = expmap_novig

    def compute(self, pars, imgarrs):
        scale = math.exp(pars['%s_logscale' % self.name].v)
        for i, (image, imgarr) in enumerate(zip(self.images, imgarrs)):
            imgkey = '%s_%s' % (self.name, image.img_id)

            v = math.exp(pars[imgkey].v) * image.pixsize_as**2 * scale
            fracvig = (lambda x: math.exp(x)/(math.exp(x)+1))(
                pars['%s_vf' % imgkey].v
            )

            out = v * (
                image.expmaps[self.expmap] * fracvig +
                image.expmaps[self.expmap_novig]*((1-fracvig)*self.vigratios[i])
            )

            imgarr += out

    def estimatePars(self, pars, images, binup=8):
        """Take images and estimate initial background values."""

        scale = math.exp(pars['%s_logscale' % self.name].v)
        for i, image in enumerate(images):
            imgkey = '%s_%s' % (self.name, image.img_id)

            fracvig = (lambda x: math.exp(x)/(math.exp(x)+1))(
                pars['%s_vf' % imgkey].v
            )

            expmap = (
                image.expmaps[self.expmap]*fracvig +
                image.expmaps[self.expmap_novig]*((1-fracvig)*self.vigratios[i])
            )

            average = N.mean((image.imagearr / expmap)[image.mask != 0])
            if average > 0:
                v = math.log(average / scale / image.pixsize_as**2)
                pars['%s_%s' % (self.name, image.img_id)].val = v

class SrcModelBase:
    """Base class for source models.

    :param name: name of model
    :param pars: dict of parameters
    :param images: list of data.Image objects
    :param cx: initial centre x coordinate
    :param cy: initial centre y coordinate
    """

    def __init__(self, name, pars, images, cx=0., cy=0.):
        self.name = name
        self.images = images

        # position of source
        pars['%s_cx' % name] = Par(cx)
        pars['%s_cy' % name] = Par(cy)

    def compute(self, pars, imgarrs):
        pass

    def prior(self, pars):
        return 0
