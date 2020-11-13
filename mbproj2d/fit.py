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

import numpy as N
import scipy.optimize
import pickle

try:
    import nlopt
except ImportError:
    nlopt = None

from . import utils
from . import fast

class Likelihood:
    """Calculate a likelihood for the model and data.

    Members:
    - prior  - prior value
    - images - list of likelihoods for each input image
    - total  - combined likelihood
    """

    def __init__(self, images, model, pars):
        self.prior = model.prior(pars)

        # do not evaluate a bad model if the prior is out of range
        if not N.isfinite(self.prior):
            imagelikes = [-N.inf]*len(images)
        else:
            modelarrs = model.compute(pars)
            imagelikes = []
            for modarr, image in zip(modelarrs, images):
                like = fast.calcPoissonLogLikelihoodMasked(
                    image.imagearr, modarr, image.mask)
                imagelikes.append(like)

        self.images = imagelikes
        self.total = self.prior + sum(imagelikes)

        if not N.isfinite(self.total):
            # force nan and inf to -inf
            self.total = -N.inf

class Fit:
    """For fitting a total model to the data."""

    def __init__(self, images, totalmodel, pars):
        self.images = images
        self.model = totalmodel
        self.pars = pars

    def printHeader(self):
        """Show line with list of thawed/non-linked params."""
        keys = ['Like', '(Prior)'] + self.pars.freeKeys()
        out = ['%12s' % k for k in keys]
        utils.uprint(' '.join(out))

    def printPars(self, like):
        """Print thawed/non-linked parameters to output on single line."""
        vals = [like.total, like.prior] + self.pars.freeVals()
        out = [('%12g' % v) for v in vals]
        utils.uprint(' '.join(out))

    def run(
            self,
            verbose=True, sigdelta=0.01, maxloops=10, maxfititer=None,
            methods=None,
            fitoptions=None,
    ):
        """
        :param verbose: show fit progress
        :param sigdelta: what is a significant change in fit statistic
        :param maxfititer: maximum number of iterations to pass to fit routine
        :param maxloops: fit a maximum number of times with improvement before exiting
        :param methods: iterate through these fitting methods to look for improvement
        :param fitoptions: options to pass to minimizer

        Default methods are
          ('LN_SBPLX', 'LN_BOBYQA'): if nlopt is installed
          ('Nelder-Mead', 'Powell'): if nlopt is not installed

        Returns (fit_like, success) where success indicates less than maximum number of iterations were done.
        """

        initlike = Likelihood(self.images, self.model, self.pars)
        showlike = [initlike.total]

        if methods is None:
            if nlopt is None:
                methods = ('Nelder-Mead', 'Powell')
            else:
                methods = ('LN_SBPLX', 'LN_BOBYQA')

        if verbose:
            self.printHeader()
            self.printPars(initlike)

        initlike = initlike.total

        def fitfunc(p, *args):
            self.pars.setFree(p)
            like = Likelihood(self.images, self.model, self.pars)
            if verbose and like.total > showlike[0]+sigdelta:
                showlike[0] = like.total
                self.printPars(like)
            return -like.total

        success = True
        fpars = self.pars.freeVals()
        for i in range(maxloops):
            if verbose:
                utils.uprint('Fitting (iteration %i)' % (i+1))

            options = {}
            if fitoptions:
                options.update(fitoptions)
            if maxfititer:
                options['maxiter'] = maxfititer

            for method in methods:
                utils.uprint(' Using method %s' % method)
                if method[:3].lower() == 'ln_':
                    # nlopt methods
                    meth = getattr(nlopt, method.upper())
                    opt = nlopt.opt(meth, len(fpars))
                    opt.set_min_objective(fitfunc)
                    opt.set_xtol_abs(1e-3)
                    lbounds, ubounds = self.pars.bounds()
                    opt.set_lower_bounds(lbounds)
                    opt.set_upper_bounds(ubounds)
                    fpars = opt.optimize(fpars)
                    flike = -opt.last_optimum_value()
                else:
                    # scipy methods
                    fitpars = scipy.optimize.minimize(
                        fitfunc, fpars, method=method, options=options)
                    fpars = fitpars.x
                    flike = -fitpars.fun

            if abs(initlike-flike) < sigdelta:
                # fit quality stayed the same
                break
            initlike = flike
        else:
            if verbose:
                utils.uprint(
                    'Exiting after maximum of %i loops' % maxloops)
            success = False

        self.pars.setFree(fpars)
        if verbose:
            utils.uprint('Done (like=%g)' % flike)
        return flike, success

    def save(self, filename):
        """Saves the current fit state as a Python pickle.

        This includes the input data, parameters and model.

        *Note*: upgrading the source code of mbproj2d or your custom
        model may prevent the saved file from being loadable
        again. Take care before relying on this for long term storage.

        :param filename: output filename

        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
