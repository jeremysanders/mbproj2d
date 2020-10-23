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
import h5py

from .fast import accreteBinImage, buildVoronoiMap
from .utils import uprint
from . import utils

class SBMaps:
    """Construct a set of maps, showing the range and residuals.

    Adaptive binning using Centroidal Voronoi Tessellations (CVT),
    following Cappellari & Copin (2003) is applied for calculating
    residuals.

    :param pars: Pars() object with parameters
    :param model: TotalModel object
    :param images: list of Image objects
    :param img_bincts: Bin images by given counts
    """

    def __init__(
            self, pars, model, images,
            img_bincts=10,
            prof_bincts=10, prof_origin=(0,0), verbose=True):

        self.verbose = verbose
        self.images = images
        self.prof_binmaps, self.prof_binradii = self._binProfiles(prof_bincts)
        self.cvt_binmaps = self._makeBinmaps(img_bincts)
        self.pars = pars
        self.model = model

    def _makeBinmaps(self, bincts):
        """Construct CVT binmaps maps for images"""
        if self.verbose:
            uprint('SBMaps: Constructing CVT maps in bands')
        binmaps = []
        for image in self.images:
            # apply bin accretion algorithm
            binmap = accreteBinImage(image.imagearr, image.mask, bincts)
            # repeatedly apply CVT

            for i in range(6):
                buildVoronoiMap(image.imagearr, binmap)

            # in above image, -1 is masked regions, and bins are
            # counted from 0. We increment this by 1 so 0 is masked,
            # so numpy binning function works ok.
            binmap[binmap<0] = -1
            binmap = binmap + 1

            binmaps.append(binmap)

        return binmaps

    def _binProfiles(self, bincts):
        """Construct bin maps for annular bins with a minimum number of counts."""

        if self.verbose:
            uprint('SBMaps: Constructing profile binning')

        binradii = []
        binmaps = []
        for image in self.images:
            # radius in pixels relative to the origin
            radius = N.fromfunction(
                lambda y,x: N.sqrt((x-image.origin[1])**2+(y-image.origin[0])**2),
                image.shape)
            # bin by pixels (use radius 0 for bad pixels)
            binmap = radius.astype(N.int32) + 1
            binmap[image.mask==0] = 0

            # get how many counts there are in each ring (dropping bad pixels)
            prof_cts = N.bincount(N.ravel(binmap), weights=N.ravel(image.imagearr))

            # now integrate up rings to achieve minimum number of counts
            outcts = 0
            outidx = 1
            lastr = 0
            binr = []
            for r in range(1, len(prof_cts)):
                outcts += prof_cts[r]
                if outcts >= bincts:
                    binmap[(binmap>lastr) & (binmap<=r)] = outidx
                    binr.append((0.5*(r+lastr+1), 0.5*(r-lastr-1)))
                    lastr = r
                    outidx += 1
                    outcts = 0

            # drop remaining annuli
            binmap[binmap>lastr] = 0

            binmaps.append(binmap)
            binradii.append(N.array(binr))

        return binmaps, binradii

    def loadChainFromFile(self, chainfname, burn=0, thin=10, randsamples=None):
        """Get list of parameter values from chain.

        :param chainfname: input chain HDF5 file
        :param burn: how many iterations to remove off input
        :param thin: discard every N entries
        :param randsample: randomly select given number of samples if set
        """

        if self.verbose:
            uprint('SBMaps: Loading chain from file '+chainfname)
        return utils.loadChainFromFile(
            chainfname, self.pars,
            burn=burn, thin=thin, randsamples=randsamples,
        )


    def _doReplay(self, chain, model_images):
        # replay chain and get lists of model profiles and images

        pars = self.pars.copy()
        chain = chain.reshape((-1, chain.shape[-1]))
        nchain = len(chain)

        modbins_cvt = [[] for _ in range(len(self.images))]
        modbins_profs_cmpts = {}

        modimgs = []
        if model_images:
            for image in self.images:
                modimgs.append(N.zeros(
                    (len(chain), image.shape[0], image.shape[1]), dtype=N.float32))

        for ichain, pararr in enumerate(chain):
            if self.verbose and ichain % 100 == 0:
                uprint('SBMaps:  %i/%i' % (ichain, nchain))

            pars.setFree(pararr)

            # want source and background components separately for profiles
            modarrs = self.model.compute_separate(pars)

            # bin total using CVT binmap
            for i in range(len(self.images)):
                modbin = N.bincount(
                    N.ravel(self.cvt_binmaps[i]), weights=N.ravel(modarrs['total'][i]))
                modbins_cvt[i].append(modbin)

            # do profiles for each component
            for cmptname in modarrs:
                for i in range(len(self.images)):

                    modbin = N.bincount(
                        N.ravel(self.prof_binmaps[i]),
                        weights=N.ravel(modarrs[cmptname][i]))

                    if cmptname not in modbins_profs_cmpts:
                        modbins_profs_cmpts[cmptname] = [[] for _ in range(len(self.images))]
                    modbins_profs_cmpts[cmptname][i].append(modbin)

            # collect model images
            if model_images:
                for i, imgarr in enumerate(modarrs['total']):
                    imgarr[self.images[i].mask==0] = N.nan
                    modimgs[i][ichain, :, :] = imgarr

        return modbins_cvt, modbins_profs_cmpts, modimgs

    def _getModImgStats(self, modimgs, percs, out):
        # get statistics for model images
        for modimg, image in zip(modimgs, self.images):
            perc = N.percentile(modimg, percs, overwrite_input=True, axis=0)

            out['%s_model_med' % image.img_id] = perc[0]
            out['%s_model_lo' % image.img_id] = perc[1]
            out['%s_model_hi' % image.img_id] = perc[2]

    def _getCVTStats(self, modbins_cvt, percs, out):
        # get statistics for CVT images
        # this means painting values back onto binmap
        for modvals, image, binmap in zip(modbins_cvt, self.images, self.cvt_binmaps):
            area = N.bincount(N.ravel(binmap)) * image.pixsize_as**2
            perc = N.percentile(modvals, percs, axis=0)
            perc[:,0] = N.nan   # masked pixels (binmap=0)
            perc /= area[N.newaxis,:] # divide by area
            modimg_med = perc[0][binmap]
            modimg_lo = perc[1][binmap]
            modimg_hi = perc[2][binmap]

            # bin data
            ncts = N.bincount(N.ravel(binmap), weights=N.ravel(image.imagearr))
            sb = ncts / area
            sb[0] = N.nan   # masked pixels
            ctsimg = sb[binmap]

            # write counts
            out['%s_CVT_data' % image.img_id] = ctsimg

            # write models
            out['%s_CVT_model_med' % image.img_id] = modimg_med
            out['%s_CVT_model_lo' % image.img_id] = modimg_lo
            out['%s_CVT_model_hi' % image.img_id] = modimg_hi

            # write fractional residuals
            out['%s_CVT_model_resid_med' % image.img_id] = ctsimg / modimg_med - 1
            out['%s_CVT_model_resid_lo' % image.img_id] = ctsimg / modimg_lo - 1
            out['%s_CVT_model_resid_hi' % image.img_id] = ctsimg / modimg_hi - 1

    def _getProfStats(self, modbins_profs_cmpts, percs, out):

        # number of pixels in each annulus
        areas = []
        for binmap, image in zip(self.prof_binmaps, self.images):
            areas.append( N.bincount(N.ravel(binmap)) * image.pixsize_as**2 )

        # now get profiles for each component
        for cmpt in modbins_profs_cmpts:

            cmpt_vals = modbins_profs_cmpts[cmpt]
            for image, profs, area in zip(self.images, cmpt_vals, areas):
                perc = N.percentile(profs, percs, axis=0)
                perc /= area[N.newaxis,:]
                out['%s_prof_%s' % (image.img_id, cmpt)] = N.column_stack((
                    perc[0][1:], perc[2][1:]-perc[0][1:], perc[1][1:]-perc[0][1:]))

        for image, binmap, area in zip(self.images, self.prof_binmaps, areas):
            cts = N.bincount(N.ravel(binmap), weights=N.ravel(image.imagearr))
            out['%s_prof_data' % image.img_id] = N.column_stack((
                (cts / area)[1:],
                ((1.0 + N.sqrt(cts+0.75))/area)[1:],
                -(N.sqrt(cts-0.25)/area)[1:],
            ))

        for image, radii in zip(self.images, self.prof_binradii):
            out['%s_prof_r' % image.img_id] = N.column_stack((
                radii[:,0], radii[:,1])) * image.pixsize_as

    def calcStats(self, chain, model_images=True, confint=68.269, h5fname=None):
        """Replay chain and calculate surface brightness statistics.

        :param chain: array of chain parameter values
        :param model_images: whether to calculate unbinned model image statistics
        :param confint: confidence interval to use
        :param h5fname: save outputs into hdf5 filename given if set

        Returns a dictionary.

        The dictionary or hdf5 file contain the following entries.
        Units are in XXXX

        $BAND_prof_$CMPT     : median profile for component $CMPT in $BAND
        $BAND_prof_total     : same as above for all model components
        $BAND_prof_data      : surface brightness in counts
        $BAND_prof_r         : centre of annuli for profiles (arcsec)

        The _$CMPT and _total profiles are 2D arrays with 3
        columns. The first is the median, the second the 1 sigma
        positive error, the third the 1 sigma negative error.

        The _r profile is a 2D array with 2 columns. The first is the
        centre of the annulus. The second is the half-width of the
        annular bin.

        $BAND_model_med          : median total model image in $BAND
        $BAND_model_lo           : 1sigma lower range of model image in $BAND
        $BAND_model_hi           : 1sigma lower range of model image in $BAND

        $BAND_CVT_model_med,_lo,_hi      : CVT-binned median(/lo/hi) model in $BAND
        $BAND_CVT_data                   : CVT-binned data in $BAND
        $BAND_CVT_model_resid_med,_lo,_hi: CVT-binned fractional residuals of data from model

        """

        if self.verbose:
            uprint('SBMaps: Calculating surface brightness stats')

        modbins_cvt, modbins_profs_cmpts, modimgs = self._doReplay(chain, model_images)

        if self.verbose:
            uprint('SBMaps:  calculating statistics')

        percs = [50, 50-confint/2, 50+confint/2]
        out = {}

        # optional statistics for models
        if model_images:
            self._getModImgStats(modimgs, percs, out)
            del modimgs

        # CVT output
        self._getCVTStats(modbins_cvt, percs, out)

        self._getProfStats(modbins_profs_cmpts, percs, out)

        # get profile statistics
        #for modvals, image, binmap, radii in zip(

        # write to hdf5 file
        if h5fname is not None:
            if self.verbose:
                uprint('SBMaps:  writing to', h5fname)
            with h5py.File(h5fname, 'w') as fout:
                for name in out:
                    fout.create_dataset(name, data=out[name], compression='gzip')
                    if out[name].ndim==2 and out[name].shape[1] in {2,3}:
                        fout[name].attrs['vsz_twod_as_oned'] = 1

        if self.verbose:
            uprint('SBMaps: Done')

        return out
