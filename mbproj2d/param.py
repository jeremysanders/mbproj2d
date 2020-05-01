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

class PriorBase:
    def calculate(self, val):
        return 0

    def __repr__(self):
        return '<PriorBase: None>'

class PriorFlat(PriorBase):
    def __init__(self, minval, maxval):
        PriorBase.__init__(self)
        self.minval = minval
        self.maxval = maxval

    def calculate(self, val):
        if self.minval <= val <= self.maxval:
            return 0
        else:
            return -N.inf

    def __repr__(self):
        return '<PriorFlat: minval=%s, maxval=%s>' % (
            repr(self.minval), repr(self.maxval))

class PriorGaussian(PriorBase):
    def __init__(self, mu, sigma):
        PriorBase.__init__(self)
        self.mu = mu
        self.sigma = sigma

    def calculate(self, val):
        if self.sigma <= 0:
            return -N.inf
        else:
            return (
                -0.5*math.log(2*math.pi)
                -math.log(self.sigma)
                -0.5*((val - self.mu) / self.sigma)**2
            )

    def __repr__(self):
        return '<PriorGaussian: mu=%s, sigma=%s>' % (
            self.mu, self.sigma)

class Param:
    """Parameter for model."""

    def __init__(
            self, val, prior=None, frozen=False, transform=None, linked=None,
            minval=-N.inf, maxval=N.inf):
        """
        :param float val: parameter value
        :param prior: prior object or None for flat prior
        :param frozen: whether to leave parameter frozen
        :param transform: function to transform value for model or 'exp' for an exp(x) scaling
        :param linked: another Param object to link this parameter to another
        :param float minval: minimum value for default flat prior
        :param float maxval: maximum value for default flat prior
        """

        self.val = val
        self.frozen = frozen

        if prior is None:
            self.prior = PriorFlat(minval, maxval)
        else:
            self.prior = prior

        if transform is None:
            self.transform = None
        elif transform == 'exp':
            self.transform = lambda x: math.exp(x)
        else:
            self.transform = transform

        self.linked = linked

    @property
    def v(self):
        """Value for using in model, after transformation or linking, if any."""

        if self.linked is None:
            val = self.val
        else:
            val = self.linked.val

        if self.transform is None:
            return val
        else:
            return self.transform(val)

    def isFree(self):
        """Is the parameter free?"""
        return self.linked is None and not self.frozen

    def calcPrior(self):
        """Calculate prior."""
        if self.linked is not None:
            return 0
        else:
            return self.prior.calculate(self.val)

    def __repr__(self):
        if self.linked is not None:
            p = [
                'linked=%s' % self.linked,
            ]
        else:
            p = [
                'val=%.3g' % self.val,
                'frozen=%s' % self.frozen,
            ]
        p.append('prior=%s' % repr(self.prior))
        if self.transform is not None:
            p.append('transform=%s' % self.transform)

        return '<Param: %s>' % (', '.join(p))

class Params(dict):
    """Parameters for a model. Based on a dict.
    """

    def numFree(self):
        """Return number of free parameters"""
        return len(self.freeKeys())

    def freeKeys(self):
        """Return sorted list of keys of parameters which are free"""
        return [key for key in sorted(self) if self[key].isFree()]

    def freeVals(self):
        """Return list of values for parameters which are free in sorted key order."""
        return [par.val for key, par in sorted(self.items()) if par.isFree()]

    def setFree(self, vals):
        """Given a list of values, set those which are free.

        Note: number of free parameters should be number of vals
        """
        i = 0
        for key in sorted(self):
            par = self[key]
            if par.isFree():
                par.val = vals[i]
                i += 1

    def calcPrior(self):
        """Return total prior of parameters."""
        return sum((par.calcPrior() for par in self.values()))

    def __repr__(self):
        # sorted repr (to match above)
        out = []
        for key in sorted(self):
            out.append('%s: %s' % (repr(key), repr(self[key])))
        return '{%s}' % (', '.join(out))
