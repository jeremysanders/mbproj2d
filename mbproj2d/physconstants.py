# Copyright (C) 2016 Jeremy Sanders <jeremy@jeremysanders.net>
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

"""Physical constants."""

# Mpc and kpc in cm and km
Mpc_km = 3.0856776e19
Mpc_cm = 3.0856776e24
kpc_cm = 3.0856776e21

# km in cm
km_cm = 1e5

# Gravitational constant (cm^3 g^-1 s^-2)
G_cgs = 6.67428e-8

# solar mass in g
solar_mass_g = 1.98892e33

# ratio of electrons to Hydrogen atoms
ne_nH = 1.2

# energy conversions
keV_erg = 1.6021765e-09
keV_K = 11.6048e6

# boltzmann constant iin erg K^-1
boltzmann_erg_K = 1.3806503e-16

# unified atomic mass constant in g
mu_g = 1.6605402e-24

# unified mass constants per electron
# 1.41 is the mean atomic mass of solar abundance (mostly H and He)
mu_e = 1.41 / ne_nH

# year in s
yr_s = 31556926

# convert total pressure in erg cm^-3 and electron density in cm^-3 to
# temperature in keV
P_keV_to_erg = keV_K * boltzmann_erg_K * (1 + 1/ne_nH)
