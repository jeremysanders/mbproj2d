# Copyright (C) 2020 Florian Kaefer
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

from .model import BackModelBase
from .ratecalc import ApecRateCalc
from .par import Par

class Sxbkg(BackModelBase):
    """
    Model for calculating soft X-ray background
    """

    def __init__(self, name, pars, images, expmap=None, NH_1022pcm2=0,
                 T_unabsorbed_keV=0.1, T_absorbed_keV=0.28):
        BackModelBase.__init__(self, name, pars, images, expmap)

        self.NH_1022pcm2 = NH_1022pcm2

        pars['%s_unabsorbed_T' % name] = Par(
            T_unabsorbed_keV, minval=0.01, maxval=1., frozen=False)
        pars['%s_unabsorbed_lognorm' % name] = Par(-13.0)

        pars['%s_absorbed_T' % name] = Par(
            T_absorbed_keV, minval=0.01, maxval=1., frozen=False)
        pars['%s_absorbed_lognorm' % name] = Par(-13.0)

        # this is for calculating the rates in each band
        self.imageRateCalcAbsorbed = {}
        self.imageRateCalcUnabsorbed = {}
        for img in images:
            args = img.rmf, img.arf, img.emin_keV, img.emax_keV
            self.imageRateCalcAbsorbed[img] = ApecRateCalc(*args, NH_1022pcm2, 0)
            self.imageRateCalcUnabsorbed[img] = ApecRateCalc(*args, 0, 0)

    def compute(self, pars, imgarrs):
        Tun = pars['%s_unabsorbed_T' % self.name].v
        Nun = math.exp(pars['%s_unabsorbed_lognorm' % self.name].v)

        Tab = pars['%s_absorbed_T' % self.name].v
        Nab = math.exp(pars['%s_absorbed_lognorm' % self.name].v)

        Z_solar = 1
        for img, imgarr in zip(self.images, imgarrs):
            # get rate for source in band
            rateAb = self.imageRateCalcAbsorbed[img].get(Tab, Z_solar, Nab)
            rateUn = self.imageRateCalcUnabsorbed[img].get(Tun, Z_solar, Nun)
            rate = rateAb + rateUn

            if self.expmap is not None:
                rate = img.expmaps[self.expmap] * rate

            imgarr += rate
