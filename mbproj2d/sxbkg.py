# Copyright (C) 2020 Florian Kaefer
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

from .model import BackModelBase
from .ratecalc import ApecRateCalc, PowerlawRateCalc, XspecModelRateCalc
from .par import Par, PriorGaussian

class Sxbkg(BackModelBase):
    """
    Model for calculating soft X-ray background

    Units are scaled according to some reasonable values
    (norms taken from UDS field)
    """

    def __init__(self, name, pars, images, expmap=None, NH_1022pcm2=0,
                 T_unabsorbed_keV=0.1, T_absorbed_keV=0.28,
                 scale_unabsorbed=1.60e-9, scale_absorbed=6.81e-11):
        BackModelBase.__init__(self, name, pars, images, expmap)

        self.NH_1022pcm2 = NH_1022pcm2
        self.scale_un = scale_unabsorbed
        self.scale_ab = scale_absorbed

        pars['%s_unabsorbed_T' % name] = Par(
            T_unabsorbed_keV, minval=0.01, maxval=1., frozen=True)
        pars['%s_unabsorbed_lognorm' % name] = Par(0, prior=PriorGaussian(0, 4))

        pars['%s_absorbed_T' % name] = Par(
            T_absorbed_keV, minval=0.01, maxval=1., frozen=True)
        pars['%s_absorbed_lognorm' % name] = Par(0, prior=PriorGaussian(0, 4))

        # this is for calculating the rates in each band
        self.imageRateCalcAbsorbed = {}
        self.imageRateCalcUnabsorbed = {}
        for img in images:
            args = img.rmf, img.arf, img.emin_keV, img.emax_keV
            self.imageRateCalcAbsorbed[img] = ApecRateCalc(*args, NH_1022pcm2, 0)
            self.imageRateCalcUnabsorbed[img] = ApecRateCalc(*args, 0, 0)

    def compute(self, pars, imgarrs):
        Tun = pars['%s_unabsorbed_T' % self.name].v
        Nun = math.exp(pars['%s_unabsorbed_lognorm' % self.name].v) * self.scale_un

        Tab = pars['%s_absorbed_T' % self.name].v
        Nab = math.exp(pars['%s_absorbed_lognorm' % self.name].v) * self.scale_ab

        Z_solar = 1
        for img, imgarr in zip(self.images, imgarrs):
            # get rate for source in band
            rateAb = self.imageRateCalcAbsorbed[img].get(Tab, Z_solar, Nab)
            rateUn = self.imageRateCalcUnabsorbed[img].get(Tun, Z_solar, Nun)
            rate = rateAb + rateUn
            rate = rate * img.pixsize_as**2

            if self.expmap is not None:
                rate = img.expmaps[self.expmap] * rate

            imgarr += rate

class Cxb(BackModelBase):
    """
    Model for calculating unresolved cosmic X-ray background
    """

    def __init__(self, name, pars, images, expmap=None, NH_1022pcm2=0, gamma=1.41, norm_scale=2.7e-10):
        """gamma = 1.41 (De Luca & Molendi 2004)
        norm_scale=11.6 photon/cm2/s/sr
        """
        BackModelBase.__init__(self, name, pars, images, expmap)

        self.NH_1022pcm2 = NH_1022pcm2
        self.norm_scale = norm_scale

        pars['%s_gamma' % name] = Par(gamma, minval=1.0, maxval=2.5, frozen=True)
        pars['%s_lognorm' % name] = Par(0, prior=PriorGaussian(0, 4))

        # this is for calculating the rates in each band
        self.imageRateCalc = {}
        for img in images:
            args = img.rmf, img.arf, img.emin_keV, img.emax_keV
            self.imageRateCalc[img] = PowerlawRateCalc(*args, NH_1022pcm2)

    def compute(self, pars, imgarrs):
        gamma = pars['%s_gamma' % self.name].v
        norm = math.exp(pars['%s_lognorm' % self.name].v) * self.norm_scale

        for img, imgarr in zip(self.images, imgarrs):
            # get rate for source in band
            rate = self.imageRateCalc[img].get(gamma, norm)

            # normalize rate by the pixel size, i.e. calculate the norm per unit area
            rate = rate * img.pixsize_as**2

            if self.expmap is not None:
                rate = img.expmaps[self.expmap] * rate

            imgarr += rate

class BackModelImage(BackModelBase):
    """Background model based on images.
    """

    def __init__(
            self, name, pars, images, backimgarrs, expmap=None,
    ):
        """A flat background model.

        :param name: name of model
        :param pars: dict of parameters
        :param images: list of data.Image objects
        :param backimgarrs: images to use as background for each Image
        """
        BackModelBase.__init__(self, name, pars, images, expmap=expmap)
        for image in images:
            pars['%s_%s_logscale' % (name, image.img_id)] = Par(0.0, frozen=True)
        pars['%s_logscale' % name] = Par(
            0.0, prior=PriorGaussian(0.0, 0.05), frozen=True)
        self.backimgarrs = backimgarrs

    def compute(self, pars, imgarrs):
        scale = math.exp(pars['%s_logscale' % self.name].v)
        for image, imgarr, backimgarr in zip(self.images, imgarrs, self.backimgarrs):
            v = math.exp(pars['%s_%s_logscale' % (self.name, image.img_id)].v)
            yw, xw = backimgarr.shape

            if self.expmap is not None:
                backimgarr = backimgarr * image.expmaps[self.expmap]

            imgarr[:yw,:xw] += (v*scale)*backimgarr

class BackModelXspecModel(BackModelBase):
    """Background model based on an xspec model(s) rate and an exposure map.

    :param name: model name
    :param images: list of Image objects
    :param xcms: list/tuple of xcm filenames
    :param scale0: scaling value to apply to rates computed from xcm model
    :param expmap: exposure map to use (if any)
    :param usearf: use ARF in computation of rates
    """

    def __init__(
            self, name, pars, images, xcms, scale0, expmap=None, usearf=True,
    ):
        BackModelBase.__init__(self, name, pars, images, expmap=expmap)
        pars['%s_logscale' % name] = Par(0.0, frozen=False)

        self.rates = []
        for img in images:
            arf = img.arf if usearf else 'none'

            rate = 0
            for xcm in xcms:
                rc = XspecModelRateCalc(
                    img.rmf, arf, img.emin_keV, img.emax_keV, xcm)
                rate += rc.get()

            # normalise by pixel area and overall scaling
            rate = rate*scale0 * img.pixsize_as**2
            self.rates.append(rate)

    def compute(self, pars, imgarrs):
        scale = math.exp(pars['%s_logscale' % self.name].v)

        for image, imgarr, rate in zip(self.images, imgarrs, self.rates):
            rate = rate * scale
            if self.expmap is not None:
                rate = rate * image.expmaps[self.expmap]
            imgarr += rate
