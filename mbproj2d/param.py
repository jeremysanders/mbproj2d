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

class ParamBase:
    """Base class for parameters."""

    # is this parameter just linked to another?
    linked = False

    def __init__(self, val, frozen=False):
        """
        :param float val: value of parameter
        :param bool frozen: whether parameter is allowed to vary
        """
        self.val = float(val)
        self.defval = self.val
        self.frozen = frozen

    def __repr__(self):
        return '<ParamBase: val=%.3g, frozen=%s>' % (
            self.val, self.frozen)

    def vout(self):
        """Return value to be used when calculating model."""
        return self.val

    def prior():
        """Log prior on parameter."""
        return 0.

class Param(ParamBase):
    """Model parameter with minimum and maximum value."""

    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        """
        :param float val: value of parameter
        :param float minval: minimum allowed value
        :param float maxval: maximum allowed value
        :param bool frozen: whether parameter is allowed to vary
        """
        ParamBase.__init__(self, val, frozen=frozen)
        self.minval = minval
        self.maxval = maxval

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)

    def prior(self):
        if self.val < self.minval or self.val > self.maxval:
            return -N.inf
        return 0.

class ParamGaussian(ParamBase):
    """Parameter with a Gaussian/normal prior."""

    def __init__(self, val, prior_mu, prior_sigma, frozen=False):
        """
        :param float val: value of parameter
        :param float prior_mu: centre of prior
        :param float prior_sigma: width of prior
        :param bool frozen: whether parameter is allowed to vary
        """
        ParamBase.__init__(self, val, frozen=frozen)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def __repr__(self):
        return '<ParamGaussian: val=%.3g, prior_mu=%.3g, prior_sigma=%.3g, frozen=%s>' % (
            self.val, self.prior_mu, self.prior_sigma, self.frozen)

    def prior(self):
        if self.prior_sigma == 0:
            return 0.

        return (
            -0.5*math.log(2*math.pi)
            -math.log(self.prior_sigma)
            -0.5*((self.val - self.prior_mu) / self.prior_sigma)**2
            )
