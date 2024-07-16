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
from astropy.coordinates import SkyCoord
import astropy.wcs
from astropy.io import fits

from .fast import accreteBinImage, buildVoronoiMap
from .utils import uprint
from . import utils

class SBMaps:
    """Construct a set of maps and/or profiles showing the data, model and residuals.

    For the maps, adaptive binning using Centroidal Voronoi
    Tessellations (CVT), following Cappellari & Copin (2003) is
    applied for calculating residuals.

    Profiles are given counts per square arcsec, unless the
    exp_correct parameter is given, which also divides by exposure.

    :param pars: Pars() object with parameters
    :param model: TotalModel object
    :param images: list of Image objects
    :param img_bincts: Bin images by given counts
    :param prof_origin: Either profile origins in arcsec (cy,cx), relative to image origin, or a SkyCoord
    :param make_profiles: Whether to make profiles
    :param make_cvtmaps: Whether to make CVT maps
    :param exp_correct: Divide counts by exposuremap with name given (e.g. "expmap") to make rate profiles/maps

    """

    def __init__(
            self, pars, model, images,
            img_bincts=10,
            prof_bincts=10, verbose=True,
            prof_origin=(0,0),
            make_profiles=True,
            make_cvtmaps=True,
            exp_correct=None,
    ):

        self.verbose = verbose
        self.images = images

        if make_profiles:
            self.prof_binmaps, self.prof_binradii = self._binProfiles(
                prof_bincts, prof_origin)
        else:
            self.prof_binmaps = self.prof_binradii = None

        if make_cvtmaps:
            self.cvt_binmaps = self._makeBinmaps(img_bincts)
        else:
            self.cvt_binmaps = None

        self.pars = pars
        self.model = model
        self.exp_correct = exp_correct

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

    def _binProfiles(self, bincts, origin):
        """Construct bin maps for annular bins with a minimum number of counts."""

        if self.verbose:
            uprint('SBMaps: Constructing profile binning')

        binradii = []
        binmaps = []
        for image in self.images:
            if isinstance(origin, SkyCoord):
                ox, oy = astropy.wcs.utils.skycoord_to_pixel(origin, image.wcs, 0)
            else:
                ox = origin[1]/image.pixsize_as + image.origin[1]
                oy = origin[0]/image.pixsize_as + image.origin[0]

            # radius in pixels relative to the origin
            radius = N.fromfunction(
                lambda y,x: N.sqrt((x-ox)**2+(y-oy)**2),
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
            if not binr:
                # this makes a zero-sized 2D array to allow slicing
                # later if there weren't enough counts to make a bin
                binr = N.zeros(shape=(0,2))

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
            if self.cvt_binmaps is not None:
                for i in range(len(self.images)):
                    weights = modarrs['total'][i]
                    if self.exp_correct is not None:
                        weights = weights / self.images[i].expmaps[self.exp_correct]

                    modbin = N.bincount(
                        N.ravel(self.cvt_binmaps[i]), weights=N.ravel(weights))
                    modbins_cvt[i].append(modbin)

            # do profiles for each component
            if self.prof_binmaps is not None:
                for cmptname in modarrs:
                    for i in range(len(self.images)):
                        weights = modarrs[cmptname][i]
                        if self.exp_correct is not None:
                            weights = weights / self.images[i].expmaps[self.exp_correct]

                        modbin = N.bincount(
                            N.ravel(self.prof_binmaps[i]),
                            weights=N.ravel(weights))

                        if cmptname not in modbins_profs_cmpts:
                            modbins_profs_cmpts[cmptname] = [[] for _ in range(len(self.images))]
                        modbins_profs_cmpts[cmptname][i].append(modbin)

            # collect model images
            if model_images:
                for i, imgarr in enumerate(modarrs['total']):
                    # imgarr[self.images[i].mask==0] = N.nan
                    modimgs[i][ichain, :, :] = imgarr

        return modbins_cvt, modbins_profs_cmpts, modimgs

    def _outputImage(self, out_images, image, name, vals):
        outhdr = None
        if image.wcs is not None:
            outhdr = image.wcs.to_header()
        hdu = fits.ImageHDU(vals, header=outhdr)
        k = f'{image.img_id}_{name}'
        hdu.header['EXTNAME'] = k
        out_images[k] = hdu

    def _getModImgStats(self, modimgs, percs, out_images):
        # get statistics for model images
        for modimg, image in zip(modimgs, self.images):
            perc = N.percentile(modimg, percs, overwrite_input=True, axis=0)
            self._outputImage(out_images, image, 'model_med', perc[0])
            self._outputImage(out_images, image, 'model_lo', perc[1])
            self._outputImage(out_images, image, 'model_hi', perc[2])

    def _getCVTStats(self, modbins_cvt, percs, out_images):
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
            ctweight = image.imagearr
            if self.exp_correct is not None:
                # make sb include exposure
                ctweight = ctweight / image.expmaps[self.exp_correct]
            ncts = N.bincount(N.ravel(binmap), weights=N.ravel(ctweight))
            sb = ncts / area
            sb[0] = N.nan   # masked pixels
            ctsimg = sb[binmap]

            # write counts
            self._outputImage(out_images, image, 'CVT_data', ctsimg)
            # write residuals
            self._outputImage(out_images, image, 'CVT_model_med', modimg_med)
            self._outputImage(out_images, image, 'CVT_model_lo', modimg_lo)
            self._outputImage(out_images, image, 'CVT_model_hi', modimg_hi)
            # write fractional residuals
            self._outputImage(out_images, image, 'CVT_model_resid_med', ctsimg/modimg_med - 1)
            self._outputImage(out_images, image, 'CVT_model_resid_lo', ctsimg/modimg_lo - 1)
            self._outputImage(out_images, image, 'CVT_model_resid_hi', ctsimg/modimg_hi - 1)

    def _getProfStats(self, modbins_profs_cmpts, percs, cols_hdf5, cols_fits):

        # routines for setting 3, 2 and 1 column entries in the output
        def setfits(img, name, val):
            tab = f'prof_{img.img_id}'
            if tab not in cols_fits:
                cols_fits[tab] = {}
            cols_fits[tab][name] = val
        def sethdf5(img, name, val):
            cols_hdf5[f'{img.img_id}_prof_{name}'] = val

        def set3(img, name, val, perr, nerr):
            setfits(img, name, val)
            setfits(img, name+'_perr', perr)
            setfits(img, name+'_nerr', nerr)
            sethdf5(img, name, N.column_stack((val, perr, nerr)))

        def set2(img, name, val, serr):
            setfits(img, name, val)
            setfits(img, name+'_serr', serr)
            sethdf5(img, name, N.column_stack((val, serr)))

        def set1(img, name, val):
            setfits(img, name, val)
            sethdf5(img, name, val)

        # number of pixels in each annulus and (optionally) exposure corrected area
        areas = []
        ctscales = []
        for binmap, image in zip(self.prof_binmaps, self.images):
            areas.append( N.bincount(N.ravel(binmap)) * image.pixsize_as**2 )
            if self.exp_correct is None:
                ctscale = 1/areas[-1]
            else:
                ctscale = 1/(
                    N.bincount(N.ravel(binmap), weights=N.ravel(image.expmaps[self.exp_correct]))
                    * image.pixsize_as**2)
            ctscales.append(ctscale)

        # now get profiles for each component
        for cmpt in modbins_profs_cmpts:
            cmpt_vals = modbins_profs_cmpts[cmpt]
            for image, profs, area in zip(self.images, cmpt_vals, areas):
                perc = N.percentile(profs, percs, axis=0)
                perc /= area[N.newaxis,:]
                val, perr, nerr = perc[0][1:], perc[2][1:]-perc[0][1:], perc[1][1:]-perc[0][1:]
                set3(image, cmpt, val, perr, nerr)

        for image, binmap, ctscale in zip(self.images, self.prof_binmaps, ctscales):
            cts = N.bincount(N.ravel(binmap), weights=N.ravel(image.imagearr))
            val = (cts * ctscale)[1:]
            perr = ((1.0 + N.sqrt(cts+0.75)) * ctscale)[1:]
            nerr = -(N.sqrt(cts-0.25) * ctscale)[1:]
            set3(image, 'data', val, perr, nerr)

        for image, radii, area in zip(self.images, self.prof_binradii, areas):
            val, serr = radii[:,0]*image.pixsize_as, radii[:,1]*image.pixsize_as
            set2(image, 'r', val, serr)
            set1(image, 'area', area[1:])

    def calcStats(self, chain, model_images=True, confint=68.269, h5fname=None, fitsfname=None):
        """Replay chain and calculate surface brightness statistics.

        :param chain: array of chain parameter values
        :param model_images: whether to calculate unbinned model image statistics
        :param confint: confidence interval to use
        :param h5fname: save outputs into hdf5 filename given if set
        :param fitsfname: save outputs into a fits filename given if set

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

        if chain.size == 0:
            raise RuntimeError('Empty chain')

        modbins_cvt, modbins_profs_cmpts, modimgs = self._doReplay(chain, model_images)

        if self.verbose:
            uprint('SBMaps:  calculating statistics')

        percs = [50, 50-confint/2, 50+confint/2]
        cols_hdf5 = {}
        cols_fits = {}
        out_images = {}

        # optional statistics for models
        if model_images:
            self._getModImgStats(modimgs, percs, out_images)
            del modimgs

        # CVT output
        if self.cvt_binmaps is not None:
            self._getCVTStats(modbins_cvt, percs, out_images)

        # profile statistics
        if self.prof_binmaps is not None:
            self._getProfStats(modbins_profs_cmpts, percs, cols_hdf5, cols_fits)

        # write to hdf5 file
        if h5fname is not None:
            if self.verbose:
                uprint('SBMaps:  writing to', h5fname)
            with h5py.File(h5fname, 'w') as fout:
                for name in cols_hdf5:
                    fout.create_dataset(name, data=cols_hdf5[name], compression='gzip')
                    if cols_hdf5[name].ndim==2 and cols_hdf5[name].shape[1] in {2,3}:
                        fout[name].attrs['vsz_twod_as_oned'] = 1
                for name in out_images:
                    fout.create_dataset(name, data=out_images[name].data, compression='gzip')

        # write to fits file
        if fitsfname is not None:
            if self.verbose:
                uprint('SBMaps:  writing to', fitsfname)

            hdus = [fits.PrimaryHDU()]
            for name in out_images:
                hdus.append(out_images[name])
            for name, vals in cols_fits.items():
                cols = []
                for cname, cvals in vals.items():
                    cols.append(fits.Column(name=cname, array=cvals, format='D'))
                hdu = fits.BinTableHDU.from_columns(cols)
                hdu.header['EXTNAME'] = name
                hdus.append(hdu)

            fits.HDUList(hdus).writeto(fitsfname, overwrite=True)

        if self.verbose:
            uprint('SBMaps: Done')

        # combine images and columns for python
        out = {}
        for name, val in out_images.items():
            out[name] = val.data
        for name, val in cols_hdf5.items():
            out[name] = val
        return out
