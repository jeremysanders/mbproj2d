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
from scipy.special import hyp2f1

from .physconstants import kpc_cm
from .par import Par
from . import utils

class Radii:
    def __init__(self, rshell_kpc, num):
        """Radii are equally-spaced with spacing rshell_kpc and num annuli/shells."""

        self.num = num
        self.rshell_kpc = rshell_kpc

        self.edges_kpc = N.arange(num+1)*rshell_kpc
        self.inner_kpc = self.edges_kpc[:-1]
        self.outer_kpc = self.edges_kpc[1:]
        self.cent_kpc = 0.5*(self.edges_kpc[1:]+self.edges_kpc[:-1])
        self.cent_logkpc = N.log(self.cent_kpc)
        self.area_kpc2 = math.pi * utils.diffSqr(self.outer_kpc, self.inner_kpc)
        self.vol_kpc3 = (4/3*math.pi)*utils.diffCube(self.outer_kpc, self.inner_kpc)

        # matrix to convert from emissivity (per kpc3) to surface
        # brightness (per kpc2). projectionVolumeMatrix produces a
        # matrix which gives the total counts per annulus, so we want
        # to divide by the annulus area.
        self.proj_matrix = (
            utils.projectionVolumeMatrix(self.edges_kpc) / self.area_kpc2[:,N.newaxis] )

    def project(self, emissivity_pkpc3):
        """Project from emissivity profile to surface brightness (per kpc2)."""
        return self.proj_matrix.dot(emissivity_pkpc3).astype(N.float32)

class ProfileBase:
    def __init__(self, name, pars):
        self.name = name

    def compute(self, pars, radii):
        """Compute profile at centres of bins, given edges."""
        return N.zeros(radii.num)

    def prior(self):
        """Return any prior associated with this profile."""
        return 0

class ProfileFlat(ProfileBase):
    def __init__(self, name, pars, defval=0., log=False):
        ProfileBase.__init__(self, name, pars)
        pars[name] = Par(defval)
        self.log = log

    def compute(self, pars, radii):
        v = pars[self.name].v
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
            pars['%s_%03i' % (name, i)] = Par(defval)
        self.rbin_edges_kpc = rbin_edges_kpc
        self.log = log

    def compute(self, radii):
        pvals = N.array([
            pars['%s_%03i' % (self.name, i)].v
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
        """Create interpolated profile between fixed values

        rcent_kpc: where to interpolate between in kpc
        """

        ProfileBase.__init__(self, name, pars)
        for i in len(rbin_edges_kpc)-1:
            pars['%s_%03i' % (name, i)] = Par(defval)
        self.rcent_logkpc = N.log(rcent_kpc)
        self.log = log

    def compute(self, pars, radii):
        pvals = N.array([
            pars['%s_%03i' % (self.name, i)].v
            for i in range(radii.num)
            ])
        vals = N.interp(radii.cent_logkpc, self.rcent_logkpc, pvals)
        if self.log:
            vals = N.exp(vals)
        return vals

def _betaprof(rin_kpc, rout_kpc, n0, beta, rc_kpc):
    """Return beta function density profile

    Calculates average density in each shell.
    """

    # this is the average density in each shell
    # i.e.
    # Integrate[n0*(1 + (r/rc)^2)^(-3*beta/2)*4*Pi*r^2, r]
    # between r1 and r2
    def intfn(r_kpc):
        return (
            4/3 * n0 * math.pi * r_kpc**3 *
            hyp2f1(3/2, 3/2*beta, 5/2, -(r_kpc/rc_kpc)**2)
        )
    nav = (intfn(rout_kpc) - intfn(rin_kpc)) / (
        4/3*math.pi * utils.diffCube(rout_kpc,rin_kpc))
    return nav

class ProfileBeta(ProfileBase):
    """Beta model.

    Parameterised by:
    logn0: log_e of n0
    beta: beta value
    logrc: log_e of rc_kpc.
    """

    def __init__(self, name, pars):
        ProfileBase.__init__(self, name, pars)
        pars['%s_logn0' % name] = Par(math.log(1e-3), minval=-14., maxval=5.)
        pars['%s_beta' % name] = Par(2/3, minval=0., maxval=4.)
        pars['%s_logrc' % name] = Par(math.log(300), minval=-2, maxval=8.5)

    def compute(self, pars, radii):
        n0 = math.exp(pars['%s_logn0' % self.name].v)
        beta = pars['%s_beta' % self.name].v
        rc_kpc = math.exp(pars['%s_logrc' % self.name].v)

        prof = _betaprof(
            radii.inner_kpc, radii.outer_kpc,
            n0, beta, rc_kpc)
        return prof

class ProfileVikhDensity(ProfileBase):
    """Density model from Vikhlinin+06, Eqn 3.

    Modes:
    'double': all components
    'single': only first component
    'betacore': only first two terms of 1st cmpt (beta, with powerlaw core)

    Densities and radii are are log base 10
    """

    def __init__(self, name, pars, mode='double'):
        ProfileBase.__init__(self, name, pars)
        self.mode = mode

        pars['%s_logn0_1' % name] = Par(math.log(1e-3), minval=-14., maxval=5.)
        pars['%s_beta_1' % name] = Par(2/3., minval=0., maxval=4.)
        pars['%s_logrc_1' % name] = Par(math.log(300), minval=-2, maxval=8.5)
        pars['%s_alpha' % name] = Par(0., minval=-1, maxval=2.)

        if mode in {'single', 'double'}:
            pars['%s_epsilon' % name] = Par(3., minval=0., maxval=5.)
            pars['%s_gamma' % name] = Par(3., minval=0., maxval=10, frozen=True)
            pars['%s_logr_s' % name] = Par(math.log(500), minval=0, maxval=8.5)

        if mode == 'double':
            pars['%s_logn0_2' % name] = Par(math.log(0.1), minval=-14., maxval=5.)
            pars['%s_beta_2' % name] = Par(0.5, minval=0., maxval=4.)
            pars['%s_logrc_2' % name] = Par(math.log(50), minval=-2, maxval=8.5)

    def compute(self, pars, radii):
        n0_1 = math.exp(pars['%s_logn0_1' % self.name].v)
        beta_1 = pars['%s_beta_1' % self.name].v
        rc_1 = math.exp(pars['%s_logrc_1' % self.name].v)
        alpha = pars['%s_alpha' % self.name].v

        r = radii.cent_kpc
        retn_sqd = (
            n0_1**2 *
            (r/rc_1)**(-alpha) / (
                (1+r**2/rc_1**2)**(3*beta_1-0.5*alpha))
            )

        if self.mode in ('single', 'double'):
            r_s = math.exp(pars['%s_logr_s' % self.name].v)
            epsilon = pars['%s_epsilon' % self.name].v
            gamma = pars['%s_gamma' % self.name].v

            retn_sqd /= (1+(r/r_s)**gamma)**(epsilon/gamma)

        if self.mode == 'double':
            n0_2 = math.exp(pars['%s_logn0_2' % self.name].v)
            rc_2 = math.exp(pars['%s_logrc_2' % self.name].v)
            beta_2 = pars['%s_beta_2' % self.name].v

            retn_sqd += n0_2**2 / (1 + r**2/rc_2**2)**(3*beta_2)

        ne = N.sqrt(retn_sqd)
        return ne
