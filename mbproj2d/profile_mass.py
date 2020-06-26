# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
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

"""Define mass profiles."""

import math
import numpy as N
import scipy.special

from .par import Par
from .physconstants import Mpc_km, G_cgs, Mpc_cm, km_cm, kpc_cm, solar_mass_g

class ProfileMassBase:
    """Base mass profile class."""

    def __init__(self, name, cosmo, pars):
        self.name = name
        self.cosmo = cosmo
        self.pars = pars

    def compute(self, pars, radii):
        """Get gravitational field and potential profiles

        :param pars: Pars() parameters objects()
        :param radii: Radii() object

        Returns: g, Phi
        """

    def prior(self, pars):
        return 0

class ProfileMassSum(ProfileMassBase):
    """Add together different profiles."""

    def __init__(self, name, cosmo, pars, cmpts):
        ProfileMassBase.__init__(self, name, cosmo, pars)
        self.cmpts = cmpts

    def compute(self, pars):
        tot_g, tot_pot = 0., 0.
        for cmpt in self.cmpts:
            g, pot = cmpt.compute(pars)
            tot_g += g
            tot_pot += pot
        return tot_g, tot_pot

    def prior(self, pars):
        return sum((prof.prior(pars) for prof in self.cmpts))

class ProfileMassNFW(ProfileMassBase):
    """NFW profile.
    Useful detals here:
    http://nedwww.ipac.caltech.edu/level5/Sept04/Brainerd/Brainerd5.html
    and Lisa Voigt's thesis

    Model parameters are nfw_logconc (log10 concentration) and
    nfw_r200_logMpc (log10 r200 in Mpc)

    """

    def __init__(self, name, cosmo, pars):
        ProfileMassBase.__init__(self, name, cosmo, pars)
        pars['%s_logconc' % name] = Par(4., minval=-4, maxval=6)
        pars['%s_r200_logMpc' % name] = Par(0., minval=-2.3, maxval=2.3)

    def compute(self, pars, radii):
        c = math.exp(pars['%s_logconc' % self.name].v)
        r200 = math.exp(pars['%s_r200_logMpc' % self.name].v)


        # relationship between r200 and scale radius
        rs_Mpc = r200 / c

        # calculate characteristic overdensity of halo (using 200
        # times critical mass density)
        delta_c = (200/3) * c**3 / (math.log(1.+c) - c/(1+c))
        # Hubble's constant at z (km/s/Mpc)
        cosmo = self.cosmo
        Hz_km_s_Mpc = cosmo.H0 * math.sqrt(
            cosmo.WM*(1.+cosmo.z)**3 + cosmo.WV )
        # critical density at redshift of halo
        rho_c = 3. * ( Hz_km_s_Mpc / Mpc_km )**2 / (8 * math.pi * G_cgs)
        rho_0 = delta_c * rho_c

        # radius relative to scale radius (centre of shell here)
        x = radii.cent_kpc * (1/(rs_Mpc*1000))

        # temporary quantities
        r_cube = (rs_Mpc * Mpc_cm)**3
        log_1x = N.log(1.+x)

        # mass enclosed within x
        mass = (4 * math.pi * rho_0) * r_cube * (log_1x - x/(1.+x))

        # gravitational acceleration
        g = G_cgs * mass / radii.cent_cm**2

        # potential
        Phi = (-4 * math.pi * rho_0 * G_cgs) * r_cube * log_1x / radii.cent_cm

        return g, Phi

class ProfileMassGNFW(ProfileMassBase):
    """Generalised NFW.

    This is an NFW with a free inner slope (alpha).

    rho(r) = rho0 / ( (r/rs)**alpha * (1+r/rs)**(3-alpha) )

    For details see Schmidt & Allen (2007)
    http://adsabs.harvard.edu/doi/10.1111/j.1365-2966.2007.11928.x

    Model parameters are gnfw_logconc (log10 concentration),
    gnfw_r200_logMpc (log10 r200 in Mpc) and gnfw_alpha (alpha
    parameter; 1 is standard NFW).

    """

    def __init__(self, name, cosmo, pars):
        ProfileMassBase.__init__(self, name, cosmo, pars)
        pars['%s_logconc' % name] = Par(4., minval=-4, maxval=6)
        pars['%s_r200_logMpc' % name] = Par(0., minval=-2.3, maxval=2.3)
        pars['%s_alpha' % name] = Par(1., minval=0., maxval=2.5)

    def compute(self, pars, radii):
        # get parameter values
        c = math.exp(pars['%s_logconc' % self.name].v)
        r200_Mpc = math.exp(pars['%s_r200_logMpc' % self.name].v)
        alpha = pars['%s_alpha' % self.name].v

        # check to make sure funny things don't happen
        alpha = max(min(alpha, 2.999), 0.)

        # overdensity relative to critical density
        phi = c**(3-alpha) / (3-alpha) * scipy.special.hyp2f1(
            3-alpha, 3-alpha, 4-alpha, -c)
        delta_c = (200/3) * c**3 / phi

        # Hubble's constant at z (km/s/Mpc)
        cosmo = self.cosmo
        Hz_km_s_Mpc = cosmo.H0 * math.sqrt(
            cosmo.WM*(1.+cosmo.z)**3 + cosmo.WV )

        # critical density at redshift of halo
        rho_c = 3 * ( Hz_km_s_Mpc / Mpc_km )**2 / (8 * math.pi * G_cgs)
        rho_0 = delta_c * rho_c

        # scale radius
        rs_cm = r200_Mpc * Mpc_cm / c

        # radius of shells relative to scale radius
        x = radii.cent_kpc * (1/(rs_Mpc*1000))

        # gravitational acceleration
        g = (
            4 * math.pi * rho_0 * G_cgs / (3-alpha) * rs_cm * x**(1-alpha) *
            scipy.special.hyp2f1(3-alpha, 3-alpha, 4-alpha, -x)
            )

        # potential
        Phi = (
            4 * math.pi * rho_0 * G_cgs / (alpha-2) * rs_cm**2 * (
                1 + -x**(2-alpha) / (3-alpha) *
                scipy.special.hyp2f1(3-alpha, 2-alpha, 4-alpha, -x) )
            )

        return g, Phi

class CmptMassKing(ProfileMassBase):
    """King potential.

    This is the modified Hubble potential, where
    rho = rho0 / (1+(r/r0)**2)**1.5

    r0 = sqrt(9*sigma**2/(4*Pi*G*rho0))

    We define rho in terms of r0 and sigma

    Model parameters are king_sigma_logkmps (sigma in log10 km/s)
    and king_rcore_logkpc (rcore in log10 kpc).

    """

    def __init__(self, name, cosmo, pars):
        ProfileMassBase.__init__(self, name, cosmo, pars)
        pars['%s_sigma_logkmps' % name] = Par(7., minval=0, maxval=11)
        pars['%s_rcore_logkpc' % name] = Par(5, minval=0, maxval=8)

    def compute(self, pars, radii):
        sigma_cmps = math.exp(pars['%s_sigma_logkmps' % self.name].v) * km_cm
        r0 = math.exp(pars['%s_rcore_logkpc' % self.name].v) * kpc_cm

        # calculate central density from r0 and sigma
        rho0 = 9*sigma_cmps**2 / (4 * math.pi * G_cgs * r0**2)
        r = radii.cent_kpc * kpc_cm

        # this often occurs below, so precalculate
        rsqrtfac = N.sqrt(r**2 + r0**2)

        g = (G_cgs/r**2)*(4*math.pi*r0**3*rho0) * (
            -r / rsqrtfac +
             N.arcsinh(r/r0))

        # taken from isothermal.nb
        phi = ( -8 * G_cgs * math.pi * (r0/r)**3 * (
                (r*N.sqrt(((r**2 + r0**2)*(-r0 + rsqrtfac))/
                          (r0 + rsqrtfac)) +
                 r0*N.sqrt(r**2 + 2*r0*(r0 - rsqrtfac)))* rho0 *
                N.arcsinh(N.sqrt(-1./2 + 0.5*N.sqrt(1 + r**2/r0**2))) ))

        return g, phi

class ProfileMassPoint(ProfileMassBase):
    """Point mass.

    Model parameter is pt_M_logMsun, which is point mass in log solar
    masses.

    """

    def __init__(self, name, cosmo, pars):
        ProfileMassBase.__init__(self, name, cosmo, pars)
        pars['%s_M_logMsun' % self.name] = Par(27., minval=20, maxval=35)

    def compute(self, pars, radii):
        mass_g = math.exp(pars['%s_M_logMsun' % self.name].v) * solar_mass_g

        r = self.radii.cent_kpc * kpc_cm
        g = G_cgs * mass_g / r**2
        phi = -G_cgs * mass_g / r
        return g, phi
