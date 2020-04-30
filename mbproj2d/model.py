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

import math
import numpy as N

from .param import Param, ParamGaussian

class TotalModel:

    def __init__(self, pars, images, objmodels=None, backmodels=None):
        self.pars = pars
        self.images = images
        self.objmodels = [] if objmodels is None else objmodels
        self.backmodels = [] if backmodels is None else backmodels

    def compute(self):
        """Compute model images given input

        Returns (objimagearrlist, bgimgarrlist)
        """

        objimgarrs = [
            N.zeros(image.shape, dtype=N.float32)
            for image in self.images
        ]
        for model in self.objmodels:
            model.compute(objimgarrs)

        bgimgarrs = [
            N.zeros(image.shape, dtype=N.float32)
            for image in self.images
        ]
        for model in self.backmodels:
            model.compute(bgimgarrs)

        return objimgarrs, bgimgarrs

    def prior(self):
        """Given parameters, compute prior."""

        total = 0
        for model in self.objmodels:
            total += model.prior()
        for model in self.backmodels:
            total += model.prior()
        return total

class BackModelBase:
    """Base background model. Does nothing."""

    def __init__(self, name, pars, images):
        self.name = name
        self.pars = pars
        self.images = images

    def compute(self, imgarrs):
        pass

    def prior(self):
        return 0

class BackModelFlat(BackModelBase):
    """Flat background model."""

    def __init__(
            self, name, pars, images,
            log=False, normarea=False,
            expmap=None
    ):
        """A flat background model.

        :param name: name of model
        :param pars: dict of parameters
        :param images: list of data.Image objects
        :param bool log: apply log scaling to value of background
        :param bool normarea: normalise background to per sq arcsec
        :param expmap: name or index of exposure map to use (if any)
        """
        BackModelBase.__init__(self, name, pars, images)
        for image in images:
            pars['%s_%s' % (name, image.imgid)] = Param(0.)
        pars['%s_scale' % name] = ParamGaussian(
            1.0, 1.0, 0.05, frozen=True)
        self.normarea = normarea
        self.log = log
        self.expmap = expmap

    def compute(self, imgarrs):
        scale = self.pars['%s_scale' % self.name].vout()
        for image, imgarr in zip(self.images, imgarrs):
            v = self.pars['%s_%s' % (self.name, image.imgid)].vout()
            if self.log:
                v = math.exp(v)
            if self.normarea:
                v *= image.pixsize_as
            if self.expmap is not None:
                v *= image.expmaps[self.expmap]
            imgarr += v*scale

class ObjModelBase:

    def __init__(self, name, pars, images, expmap=None, cx=0., cy=0.):
        """
        :param name: name of model
        :param pars: dict of parameters
        :param images: list of data.Image objects
        :param expmap: name or index of exposure map to use (if any)
        :param cx: initial centre x coordinate
        :param cy: initial centre y coordinate
        """

        self.name = name
        self.pars = pars
        self.images = images
        self.expmap = expmap

        # position of source
        pars['%s_cx' % name] = Param(cx)
        pars['%s_cy' % name] = Param(cy)

    def compute(self, imgarrs):
        pass

    def prior(self):
        return 0
