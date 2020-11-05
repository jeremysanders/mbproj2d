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

from .model import SrcModelBase
from .ratecalc import PowerlawRateCalc
from .par import Par

class PointBase(SrcModelBase):
    def __init__(self, name, pars, images, NH_1022pcm2=0., cx=0., cy=0.):
        SrcModelBase.__init__(self, name, pars, images, cx=cx, cy=cy)
        self.NH_1022pcm2 = NH_1022pcm2

class PointPowerlaw(PointBase):
    """Powerlaw model at a particular position.

    gamma is a fitting parameter (default fixed)
    """

    def __init__(self, name, pars, images, NH_1022pcm2=0., cx=0., cy=0., gamma=1.7):
        PointBase.__init__(
            self, name, pars, images, NH_1022pcm2=NH_1022pcm2, cx=cx, cy=cy)

        pars['%s_gamma' % name] = Par(gamma, minval=1.0, maxval=2.5, frozen=True)
        pars['%s_lognorm' % name] = Par(-13.0)

        # this is for calculating the rates in each band
        self.imageRateCalc = {}
        for img in images:
            self.imageRateCalc[img] = PowerlawRateCalc(
                img.rmf, img.arf, img.emin_keV, img.emax_keV, NH_1022pcm2)

    def compute(self, pars, imgarrs):
        cy_as = pars['%s_cy' % self.name].v
        cx_as = pars['%s_cx' % self.name].v
        gamma = pars['%s_gamma' % self.name].v
        norm = math.exp(pars['%s_lognorm' % self.name].v)

        for img, imgarr in zip(self.images, imgarrs):
            # get rate for source in band
            rate = self.imageRateCalc[img].get(gamma, norm)

            # split flux into 4 bins using linear interpolation
            # this makes the fitting more robust if a source doesn't jump between pixels
            cy = cy_as*img.invpixsize + img.origin[0]
            cx = cx_as*img.invpixsize + img.origin[1]
            cxi = int(cx)
            cyi = int(cy)
            yfrac = cy-cyi
            xfrac = cx-cxi
            yw, xw = imgarr.shape

            if 0<=cyi<yw and 0<=cxi<xw:
                imgarr[cyi, cxi] += rate*(1-yfrac)*(1-xfrac)
            if 0<=cyi+1<yw and 0<=cxi<xw:
                imgarr[cyi+1, cxi] += rate*yfrac*(1-xfrac)
            if 0<=cyi<yw and 0<=cxi+1<xw:
                imgarr[cyi, cxi+1] += rate*(1-yfrac)*xfrac
            if 0<=cyi+1<yw and 0<=cxi+1<xw:
                imgarr[cyi+1, cxi+1] += rate*yfrac*xfrac
