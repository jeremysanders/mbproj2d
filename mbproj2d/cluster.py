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

from .model import ObjModelBase

class ClusterNonHydro(ObjModelBase):

    def __init__(
            self, name, pars, images,
            cosmo=None,
            NH_1022cm2=0.,
            ne_prof=None, T_prof=None, Z_prof=None,
            expmap=None,
            cx=0., cy=0.
    ):
        ObjModelBase.__init__(
            self, name, pars, images, expmap=expmap, cx=cx, cy=cy)

        self.cosmo = cosmo
        self.NH_1022pcm2 = NH_1022pcm2
        self.ne_prof = ne_prof
        self.T_prof = T_prof
        self.Z_prof = Z_prof
