import time
import os.path

import numpy as N
import h5py

from .forkparallel import ForkQueuePool
from .fit import Likelihood

try:
    import emcee
except ImportError:
    emcee = None

try:
    import zeus
except ImportError:
    zeus = None

class MCMCStore:
    """Store chain in file.

    :param filename: name of file
    :param Pars pars: parameters
    :param int nwalkers: number of walkers
    :param int thin: thin entries written to file by factor
    :param float flush_time: how often to flush output to file (s)
    """

    def __init__(self, filename, pars, nwalkers, thin=1, flush_time=300):
        self.filename = filename
        self.pars = pars
        self.nwalkers = nwalkers
        self.thin = thin
        self.flush_time = flush_time

        self.last_time = time.time()
        self.nrows = 0
        self.last_pos = None
        self.npars = pars.numFree()
        self.thin_ct = 0

        self.pars_buf = []
        self.like_buf = []

        if os.path.exists(filename):
            self.openExisting()
        else:
            self.createNew()

    def numRows(self):
        """Get number of rows (in store + to write)."""
        return self.nrows

    def numSteps(self):
        """Get number of steps processed."""
        return self.nrows * self.thin

    def lastPos(self):
        """Return last point written to chain."""
        return self.last_pos

    def add(self, par_arr, like_arr):
        """Add to chain.

        :param par_arr: numpy array of parameter values (nwalkers*npars)
        :param like_arr: numpy array of likelihoods (nwalkers)
        """

        # do thinning
        self.thin_ct += 1
        if self.thin_ct < self.thin:
            return
        self.thin_ct = 0

        self.pars_buf.append(N.array(par_arr))
        self.like_buf.append(N.array(like_arr))
        self.nrows += 1
        self.last_pos = par_arr

        this_time = time.time()
        if this_time >= self.last_time + self.flush_time:
            self.flush()
            self.last_time = this_time

    def openExisting(self):
        """Open an existing file."""
        pass

    def createNew(self):
        """Create a new output file."""
        pass

    def flush(self):
        """Update output file."""
        pass

class MCMCStoreHDF5(MCMCStore):
    """Store chain in HDF5 file."""

    def openExisting(self):
        with h5py.File(self.filename, 'r') as fin:
            f_names = [x.decode('utf-8') for x in fin['thawed_params']]
            f_dims = fin['chain'].shape
            self.nrows = f_dims[0]
            if self.nrows > 0:
                self.last_pos = N.array(
                    fin['chain'][-1,:,:]).astype(N.float64)

        if f_names != self.pars.freeKeys():
            raise RuntimeError(
                f"Inconsistent parameter names in file {self.filename}")
        if f_dims[1] != self.nwalkers:
            raise RuntimeError(
                f"Inconsistent number of walkers in file {self.filename}")
        if f_dims[2] != self.npars:
            raise RuntimeError(
                f"Inconsistent number of parameters in file {self.filename}")

    def createNew(self):
        """Create a new file for the output."""
        with h5py.File(self.filename, 'w') as fout:
            fout['thawed_params'] = N.array([
                x.encode('utf-8') for x in self.pars.freeKeys()])
            fout.create_dataset(
                'chain',
                (0,self.nwalkers,self.npars),
                maxshape=(None,self.nwalkers,self.npars),
                compression=True,
                shuffle=True,
                dtype='f4',
            )
            fout.create_dataset(
                'likelihood',
                (0,self.nwalkers),
                maxshape=(None,self.nwalkers),
                compression=True,
                shuffle=True,
                dtype='f4',
            )

    def flush(self):
        """Update output file."""
        if len(self.pars_buf) == 0:
            return

        pars_buf = N.array(self.pars_buf)
        like_buf = N.array(self.like_buf)

        with h5py.File(self.filename, 'r+') as fout:
            chain = fout['chain']
            chain.resize((self.nrows,self.nwalkers,self.npars))
            chain[-len(self.pars_buf):,:,:] = pars_buf
            self.pars_buf.clear()
            like = fout['likelihood']
            like.resize((self.nrows,self.nwalkers))
            like[-len(self.like_buf):,:] = like_buf
            self.like_buf.clear()

class MCMCSampler:
    """Base MCMC sampler class."""

    def __init__(self, fit):
        self.fit = fit

    def getLikeFunc(self):
        """Return function to evaluate likelihood."""
        mcmcpars = self.fit.pars.copy()
        def likefunc(parvals):
            mcmcpars.setFree(parvals)
            like = Likelihood(self.fit.images, self.fit.model, mcmcpars)
            return like.total
        return likefunc

    def makePool(self, nprocesses, likefunc):
        """Make parallel Pool.

        :param int nprocesses: number of processes (<=1 means no fork pool)
        """
        return None if nprocesses <= 1 else ForkQueuePool(
            likefunc, nprocesses)

    def createWalkerP0(self, nwalkers, spread):
        """Helper to create starting values for walkers.

        :param int nwalkers: number of walkers
        :param float spread: Gaussian sigma to modify parameters
        """

        fit = self.fit
        pars = fit.pars.copy()
        pars_arr = N.array(pars.freeVals())
        if not N.all(N.isfinite(pars_arr)):
            raise RuntimeError("Parameter values not finite")
        if not N.isfinite(Likelihood(fit.images, fit.model, pars).total):
            raise RuntimeError("Initial likelihood not finite")

        # create enough parameters with finite likelihoods
        p0 = []
        ct = 0
        while len(p0) < nwalkers:
            if ct / 1000 > nwalkers:
                # avoid running for ever in case of a likelihood issue
                raise RuntimeError("Could not create enough parameters with a finite likelihood")
            p = N.random.normal(pars_arr, self.initspread)
            pars.setFree(p)
            like = Likelihood(fit.images, fit.model, pars).total
            if N.isfinite(like):
                p0.append(p)
            ct += 1

        return p0

    def sample(self, store, nsteps, progress=True):
        """Do MCMC sampling.
        """
        pass

    def sampleMinSteps(self, store, nsteps, progress=True):
        """Sample to MCMCStore to ensure at least nsteps of samples.

        If there are less than nsteps, the sampling is continued.

        :param store: MCMCStore object
        :param int nsteps: number of steps to ensure in output (before thinning)

        Returns number of steps done.
        """

        delta = nsteps - store.numSteps()
        if delta > 0:
            self.sample(store, delta, progress=progress)
            return delta
        return 0

class MCMCSamplerEmcee(MCMCSampler):
    """Sample fit using Emcee.

    :param fit: Fit object
    :param int nwalkers: Number of walkers
    :param int nprocesses: Number of processes to use
    :param float initspread: Spread to add to initial parameters
    """

    def __init__(self, fit, nwalkers, nprocesses=1, initspread=0.01):
        MCMCSampler.__init__(self, fit)
        self.nwalkers = nwalkers
        self.nprocesses = nprocesses
        self.npars = fit.pars.numFree()
        self.initspread = initspread

    def sample(self, store, nsteps, progress=True):
        """Do MCMC sampling.

        :param store: MCMCStore object
        :param int nsteps: number of steps to add to chain
        :param progress: show progress bar
        """

        assert store.nwalkers == self.nwalkers

        likefunc = self.getLikeFunc()
        pool = self.makePool(self.nprocesses, likefunc)
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.npars,
            likefunc,
            pool=pool,
        )

        p0 = store.lastPos()
        if p0 is None:
            p0 = self.createWalkerP0(self.nwalkers, self.initspread)

        for result in sampler.sample(
                p0,
                iterations=nsteps,
                store=False,
                progress=progress):
            store.add(result.coords, result.log_prob)
        store.flush()
