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

try:
    import emcee
except ImportError:
    emcee = None

try:
    import zeus
except ImportError:
    zeus = None

import h5py
import numpy as N

from . import forkparallel
from .fit import Likelihood
from . import utils

class MCMC:
    """
    Handles MCMC analysis of Fit

    :param Fit fit: Fit object to use for mcmc
    :param int walkers: number of walkers to use
    :param int processes: number of simultaneous processes to compute likelihoods
    :param float initspread: random Gaussian width added to create initial parameters
    :param moves: moves parameter to emcee.EnsembleSampler
    :param verbose: whether to write progress
    :param sampler: 'emcee' or 'zeus'
    """

    def __init__(
            self, fit,
            nwalkers=50, processes=1, initspread=0.01, moves=None, verbose=True,
            sampler='emcee',
    ):

        self.fit = fit
        self.nwalkers = nwalkers
        self.numpars = fit.pars.numFree()
        self.initspread = initspread
        self.verbose = verbose
        self.sampler_mode = sampler

        mcmcpars = fit.pars.copy()

        def likefunc(parvals):
            mcmcpars.setFree(parvals)
            like = Likelihood(fit.images, fit.model, mcmcpars)
            return like.total

        # create a pool if more than one process
        pool = None if processes <= 1 else forkparallel.ForkQueuePool(
            likefunc, processes)

        # for doing the mcmc sampling
        if sampler == 'emcee':
            if emcee is None:
                raise RuntimeError('emcee module not installed')
            self.sampler = emcee.EnsembleSampler(
                nwalkers,
                self.numpars,
                likefunc,
                pool=pool,
                moves=moves
            )
        elif sampler == 'zeus':
            if zeus is None:
                raise RuntimeError('zeus module not installed')
            self.sampler = zeus.EnsembleSampler(
                nwalkers,
                self.numpars,
                likefunc,
                pool=pool,
            )
        else:
            raise RuntimeError('Unknown sampler')

        # starting point
        self.pos0 = None

        # header items to write to output file
        self.header = {
            'burn': 0,
        }

    def verbprint(self, *args, **argsv):
        """Output if verbose."""
        if self.verbose:
            utils.uprint(*args, **argsv)

    def _generateInitPars(self):
        """Generate initial set of parameters from fit."""

        newpars = self.fit.pars.copy()
        thawedpars = N.array(newpars.freeVals())
        assert N.all(N.isfinite(thawedpars))

        # create enough parameters with finite likelihoods
        p0 = []
        while len(p0) < self.nwalkers:
            p = N.random.normal(0, self.initspread, size=self.numpars) + thawedpars
            newpars.setFree(p)
            like = Likelihood(self.fit.images, self.fit.model, newpars).total
            if N.isfinite(like):
                p0.append(p)

        return p0

    def _innerburn(self, length, autorefit, minfrac, minimprove):
        """Returns False if new minimum found and autorefit is set"""

        bestfit = None
        like = Likelihood(
            self.fit.images, self.fit.model, self.fit.pars).total
        bestlike = initlike = like
        p0 = self._generateInitPars()

        # record period
        self.header['burn'] = length

        # iterate over burn-in period
        for i, result in enumerate(self.sampler.sample(
                p0, iterations=length, store=False)):

            if i % 10 == 0:
                self.verbprint(' Burn %i / %i (%.1f%%)' % (
                    i, length, i*100/length))

            self.pos0 = result.coords
            lnlike = result.log_prob

            # new better fit
            if lnlike.max()-bestlike > minimprove:
                bestlike = lnlike.max()
                maxidx = lnlike.argmax()
                bestfit = self.pos0[maxidx]

            # abort if new minimum found
            if ( autorefit and i>length*minfrac and
                 bestfit is not None ):

                self.verbprint(
                    '  Restarting burn as new best fit has been found '
                    ' (%g > %g)' % (bestlike, initlike) )
                self.fit.pars.setFree(bestfit)
                self.sampler.reset()
                return False

        self.sampler.reset()
        return True

    def _innerburnzeus(self, length):
        """Do burn-in for zeus sampler."""

        # record period
        self.header['burn'] = length

        p0 = self._generateInitPars()
        self.sampler.run(p0, length)
        self.pos0 = self.sampler.get_chain()[-1,:,:]
        self.sampler.reset()
        return True

    def burnIn(self, length, autorefit=True, minfrac=0.2, minimprove=0.01):
        """Burn in, restarting fit and burn if necessary.

        :param bool autorefit: refit position if new minimum is found during burn in
        :param float minfrac: minimum fraction of burn in to do if new minimum found
        :param float minimprove: minimum improvement in fit statistic to do a new fit
        """

        self.verbprint('Burning in')

        if self.sampler_mode == 'emcee':
            while not self._innerburn(length, autorefit, minfrac, minimprove):
                self.verbprint('Restarting, as new minimum found')
                self.fit.run()
        elif self.sampler_mode == 'zeus':
            self._innerburnzeus(length)

    def run(self, length, progress=True):
        """Run chain.

        :param int length: length of chain
        """

        self.verbprint('Sampling')
        self.header['length'] = length

        # initial parameters
        if self.pos0 is None:
            self.verbprint(' Generating initial parameters')
            p0 = self._generateInitPars()
        else:
            self.verbprint(' Starting from end of burn-in position')
            p0 = self.pos0

        self.verbprint(' doing sampling')
        self.sampler.run_mcmc(p0, nsteps=length, progress=progress)

        self.verbprint('Done')

    def save(self, outfilename, thin=1, discard=0):
        """Save chain to HDF5 file.

        :param str outfilename: output hdf5 filename
        :param int thin: save every N samples from chain
        :param int discard: discard first N samples
        """

        self.header['thin'] = thin

        self.verbprint('Saving chain to', outfilename)
        with h5py.File(outfilename, 'w') as f:
            # write header entries
            for h in sorted(self.header):
                f.attrs[h] = self.header[h]

            # write list of parameters which are thawed
            f['thawed_params'] = N.array([
                x.encode('utf-8') for x in self.fit.pars.freeKeys()])

            # output chain
            f.create_dataset(
                'chain',
                data=self.sampler.get_chain(
                    thin=thin, discard=discard).astype(N.float32),
                compression=True, shuffle=True)
            # likelihoods for each walker, iteration
            f.create_dataset(
                'likelihood',
                data=self.sampler.get_log_prob(
                    thin=thin, discard=discard).astype(N.float64),
                compression=True,
                shuffle=True
            )
            if self.sampler_mode == 'emcee':
                # acceptance fraction
                f['acceptfrac'] = self.sampler.acceptance_fraction.astype(N.float32)
            # last position in chain
            f['lastpos'] = self.sampler.get_chain()[-1,:,:]

        self.verbprint('Done')
