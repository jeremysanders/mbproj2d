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

import math
import numpy as N

from .par import Par, PriorGaussian
from . import utils
from . import ratecalc

class TotalModel:
    """Combined model for data."""

    def __init__(
            self, pars, images, src_models=None, src_expmap=None,
            back_models=None):
        """
        :param pars: Pars object (currently unused)
        :param images: list of Image objects
        :param src_models: list of source models
        :param src_expmap: name of exposure map to use for sources
        :param back_models: list of background models
        """

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

class BackModelBase:
    """Base background model. Does nothing."""

    def __init__(self, name, pars, images, expmap=None):
        """
        :param name: name of model
        :param pars: Pars() object
        :param images: list of Image() objects
        :param expmap: exposure map name to lookup in Image
        """

        self.name = name
        self.images = images
        self.expmap = expmap

    def compute(self, pars, imgarrs):
        pass

    def prior(self, pars):
        return 0

class BackModelFlat(BackModelBase):
    """Flat background model."""

    def __init__(
            self, name, pars, images,
            log=False, normarea=False,
            defval=0.,
            expmap=None
    ):
        """A flat background model.

        :param name: name of model
        :param pars: dict of parameters
        :param images: list of data.Image objects
        :param bool log: apply log scaling to value of background
        :param bool normarea: normalise background to per sq arcsec
        :param defval: default parameter
        :param expmap: name or index of exposure map to use (if any)
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
                v = math.exp(v)
            if self.normarea:
                v *= image.pixsize_as
            if self.expmap is not None:
                v *= image.expmaps[self.expmap]
            imgarr += v*scale

class SrcModelBase:
    """Base class for source models."""

    def __init__(self, name, pars, images, cx=0., cy=0.):
        """
        :param name: name of model
        :param pars: dict of parameters
        :param images: list of data.Image objects
        :param cx: initial centre x coordinate
        :param cy: initial centre y coordinate
        """

        self.name = name
        self.images = images

        # position of source
        pars['%s_cx' % name] = Par(cx)
        pars['%s_cy' % name] = Par(cy)

    def compute(self, pars, imgarrs):
        pass

    def prior(self, pars):
        return 0
