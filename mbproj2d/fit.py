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

from . import utils
from . import fast

class Likelihood:
    """A likelihood for the model and data."""

    def __init__(self, images, model, pars):
        self.prior = model.prior(pars)

        modelarrs = model.compute(pars)
        imagelikes = []
        for modarr, image in zip(modelarrs, images):
            like = fast.calcPoissonLogLikelihoodMasked(
                image.imagearr, modarr, image.mask)
            imagelikes.append(like)

        self.images = imagelikes
        self.total = self.prior + sum(imagelikes)

class Fit:
    """For fitting a total model to the data."""

    def __init__(self, images, totalmodel, pars):
        self.images = images
        self.model = totalmodel
        self.pars = pars

    def printHeader(self):
        keys = ['Like', 'Prior'] + self.pars.freeKeys()
        out = ['%11s' % k for k in keys]
        utils.uprint(' '.join(out))

    def printPars(self, like):
        """Print parameters to output."""
        vals = [like.total, like.prior]
        out = [('%11g' % v.val) for k, v in sorted(self.pars.items())]
        utils.uprint(' '.join(out))

    def doFit(self, verbose=True):
        initpars = self.pars.freeVals()

        initlike = self.computeLikelihood()
        minlike = [initlike.total]

        if verbose:
            self.printHeader()
            self.printPars(initlike)

        def fitfunc(p):
            self.pars.setFree(p)
            like = Likelihood(self.images, self.model, self.pars)
            if like.total > minlike[0]:
                minlike[0] = like.total
                if verbose:
                    self.printPars(like)
            return -like.total
