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

from .physconstants import kpc_cm
from .param import Param
from .utils import diff_cube

class Radii:
    def __init__(self, edges_kpc):
        self.num = len(edges_kpc)-1
        self.edges_kpc = edges_kpc
        self.inner_kpc = edges_kpc[:-1]
        self.outer_kpc = edges_kpc[1:]
        self.cent_kpc = 0.5*(edges_kpc[1:]-edges_kpc[:-1])
        self.cent_logkpc = N.log(self.cent_kpc)

        self.vol_kpc3 = (4/3*math.pi)*diff_cube(self.outer_kpc, self.inner_kpc)

class ProfileBase:
    def __init__(self, name, pars):
        self.name = name
        self.pars = pars

    def compute(self, radii):
        """Compute profile at centres of bins, given edges."""
        return N.zeros(radii.num)

    def prior(self):
        """Return any prior associated with this profile."""
        return 0

class ProfileFlat(ProfileBase):
    def __init__(self, name, pars, defval=0., log=False):
        ProfileBase.__init__(self, name, pars)
        pars[name] = Param(defval)
        self.log = log

    def compute(self, redges_kpc):
        v = self.pars[self.name].vout()
        if self.log:
            v = math.exp(v)
        return N.full(radii.num, v)

class ProfileBinned(ProfileBase):
    def __init__(self, name, pars, rbin_edges_kpc, defval=0., log=False):
        """Create binned profile.

        rbin_edges_kpc: array of bin edges, kpc
        defval: default value
        log: where to apply exp to output.
        """

        ProfileBase.__init__(self, name, pars)
        for i in len(rbin_edges_kpc)-1:
            pars['%s_%03i' % (name, i)] = Param(defval)
        self.rbin_edges_kpc = rbin_edges_kpc
        self.log = log

    def compute(self, radii):
        pvals = N.array([
            self.pars['%s_%03i' % (self.name, i)].vout()
            for i in range(radii.num)
            ])
        if self.log:
            pvals = N.exp(vals)
        idx = N.searchsorted(self.rbin_edges_kpc[1:], radii.cent_kpc)
        idx = N.clip(idx, 0, len(pvals)-1)

        # lookup bin for value
        outvals = pvals[idx]
        # mark values outside range as nan
        outvals[radii.outer_kpc < rbin_edges_kpc[0]] = N.nan
        outvals[radii.inner_kpc > rbin_edges_kpc[-1]] = N.nan

        return outvals

class ProfileInterpol(ProfileBase):

    def __init__(self, name, pars, rcent_kpc, defval=0., log=False):
        """Create interpolated profile

        rcent_kpc: where to interpolate between in kpc
        """

        ProfileBase.__init__(self, name, pars)
        for i in len(rbin_edges_kpc)-1:
            pars['%s_%03i' % (name, i)] = Param(defval)
        self.rcent_logkpc = N.log(rcent_kpc)
        self.log = log

    def compute(self, radii):
        pvals = N.array([
            self.pars['%s_%03i' % (self.name, i)].vout()
            for i in range(radii.num)
            ])
        vals = N.interp(radii.cent_logkpc, self.rcent_logkpc, pvals)
        if self.log:
            vals = N.exp(vals)
        return vals
