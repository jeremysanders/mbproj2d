#!/usr/bin/env python3

# eFEDS example cluster fit

import math
import os
import os.path

from astropy.io import fits
import numpy as np

import mbproj2d as mb

# which energy bands to use (eV)
bands = [
    (300, 600), (600, 1100), (1100, 1600), (1600, 2200),
    (2200, 3500), (3500, 5000), (5000, 7000),
]

# HDUs in PSF calibration file for each energy band
bandPSFHDUs = [
    0, 1, 2, 2,
    3, 4, 5,
]

def loadImages(name):
    """Load images from fits files.

    Here we measure coordinates relative to origin (ra, dec)
    """

    inimgdir = os.path.join('images', name)
    inrspdir = os.path.join('responses', name)

    rmf = os.path.join(inrspdir, 'spec_020_RMF_00001.fits')
    arf = os.path.join(inrspdir, 'spec_020_ARF_00001.fits')
    psf_fname = 'calib/tm1_2dpsf_190220v03.fits.gz'

    # for each band we load the image, exposure map and PSF
    images = []
    for band, psfhdu in zip(bands, bandPSFHDUs):
        print('Loading', band)

        # name for band (e.g., '0300_1000')
        bandname = '%04i_%04i' % (band[0],band[1])

        # note: two exposure maps here - vignetted and unvignetted
        # (note exposure maps have units of seconds)
        img = mb.imageLoad(
            '%s/img_%s.fits.gz' % (inimgdir, bandname),
            '%s/expmap_norm_%s.fits.gz' % (inimgdir, bandname),
            exp_novig_fname='%s/expmap_norm_novig_%s.fits.gz' % (inimgdir, bandname),
            rmf=rmf, arf=arf,
            emin_keV=band[0]/1000, emax_keV=band[1]/1000,
            mask_fname='%s/mask.fits.gz' % inimgdir,
            pix_origin=(0,0),
            psf=mb.PSFLoad(psf_fname, psfhdu),
            pad=64,  # zero pad for convolution (wraps otherwise)
        )

        images.append(img)

    return images

def writeModelImages(fname, totmodel, pars):
    '''Write model in each band into 3D fits file.'''
    img_3d = np.array(totmodel.compute(pars))
    hdulist = fits.HDUList(fits.PrimaryHDU(img_3d))
    hdulist.writeto(fname, overwrite=True)

def fit():

    name = 'eFEDS_Example'

    # output filenames
    suffix = 'iso'
    fitdir = 'fits/%s' % name
    try:
        os.makedirs(fitdir)
    except OSError:
        pass

    # output filenames
    parfname = os.path.join(fitdir, 'pars_%s.pickle' % suffix)
    chainfname = os.path.join(fitdir, 'chain_%s.hdf5' % suffix)
    physfname = os.path.join(fitdir, 'phys_%s.hdf5' % suffix)
    modimgfname = os.path.join(fitdir, 'img_%s.fits' % suffix)
    sbfname = os.path.join(fitdir, 'sb_%s.hdf5' % suffix)

    # load images from fits files
    images = loadImages(name)

    # initial parameters
    pars = mb.Pars()

    # metallicity profile (flat, fixed)
    Z_prof = mb.ProfileFlat('Z', pars, defval=0.3)
    pars['Z'].frozen = True

    # clusters to fit simultaneously
    cl_ra = [140.3384, 140.0918]   # RA
    cl_dec = [3.2906, 3.0185] # Dec
    cl_z = [0.33, 0.48] # redshift
    cl_idx = [1, 2] # index (in parameter names)

    # Galactic column (HI in 10^22)
    NH_1022pcm2 = 0.035

    cluster_cmpts = []
    for idx, ra, dec, z in zip(cl_idx, cl_ra, cl_dec, cl_z):
        # centre of the cluster in pixels on input image
        xpos, ypos = images[0].wcs.all_world2pix(ra, dec, 0)

        # cosmology
        cosmo = mb.Cosmology(z)

        # temperature profile (flat, in log T)
        T_prof = mb.ProfileFlat(
            'T_log_%i' % idx, pars,
            defval=math.log(3),
            log=True,
            minval=math.log(0.06),
            maxval=math.log(60)
        )

        # density profile (this is a single beta-component Vikhlinin model)
        ne_prof = mb.ProfileVikhDensity('ne_%i' % idx, pars, mode='single')

        # non-hydrostatic model
        cluster = mb.ClusterNonHydro(
            'cl_%i' % idx,  # name
            pars,
            images,
            cosmo=cosmo,
            NH_1022pcm2=NH_1022pcm2,
            ne_prof=ne_prof,
            T_prof=T_prof,
            Z_prof=Z_prof,
        )

        # fix cluster position (optional!). Note: positions are
        # relative to the origin of the input images, and are in
        # arcsec
        xp = pars['cl_%i_cx' % idx]
        xp.val = xpos * images[0].pixsize_as
        xp.frozen = True
        yp = pars['cl_%i_cy' % idx]
        yp.val = ypos * images[0].pixsize_as
        yp.frozen = True

        cluster_cmpts.append(cluster)

    # flat background model in each energy band
    backmodel = mb.BackModelFlat(
        'bg', pars, images, log=True, defval=-15, normarea=True,
        expmap='expmap')

    # this is a more sophisticated background - this is a mixture of vignetted and unvignetted flat components
    # backmodel = mb.BackModelVigNoVig('bg', pars, images, defval=-15)

    # combine source models and background models to make total model
    totmod = mb.TotalModel(
        pars,
        images,
        src_expmap='expmap',
        src_models=cluster_cmpts,    # add more here or pt sources
        back_models=[backmodel],
    )

    fit = mb.Fit(images, totmod, pars)

    # load initial parameters if saved (this is a python pickle file)
    if os.path.exists(parfname):
        print('loading', parfname)
        pars.load(parfname)
    else:
        # find best fitting parameters
        pars.write()
        fit.run(maxloops=1)
        pars.write()
        # save parameters
        pars.save(parfname)

    if not os.path.exists(modimgfname):
        # writes the best-fitting model as a 3D fits file (one slice per band)
        print("Write model image")
        writeModelImages(modimgfname, totmod, pars)

    if not os.path.exists(chainfname):
        nwalkers = 3*pars.numFree()
        if nwalkers % 2 != 0:
            nwalkers += 1  # make even

        # do MCMC on parameters. The output is written to a HDF5 fits
        # containing the chain values and likelihoods
        print("Starting MCMC")
        nprocesses = 8
        mcmc = mb.MCMC(
            fit,
            processes=nprocesses,
            nwalkers=nwalkers)
        mcmc.run(1000)
        mcmc.save(chainfname, discard=500)

    if not os.path.exists(sbfname):
        # get profile and model image residuals compared to data
        # note that the origin for profiles is fixed to the coordinate given
        sbmaps = mb.SBMaps(
            pars, totmod, images,
            prof_origin=(pars['cl_1_cy'].v, pars['cl_1_cx'].v),
        )
        # h5fname is optional (by default returns info as dict)
        output = sbmaps.calcStats(
            mb.loadChainFromFile(chainfname, pars, randsamples=1000),
            h5fname=sbfname,
        )

    if not os.path.exists(physfname):
        # compute physical quantities for the first cluster
        cluster0 = cluster_cmpts[0]

        # convert chain to physical profiles for this cluster model
        phys = mb.Phys(pars, cluster0, rate_rmf=images[0].rmf, rate_arf=images[0].arf)

        # Take 1000 random samples from the chain and calculate the median physical
        # profiles and 1-sigma range.

        # The output file is a HDF5 with a set of profiles (each profile is stored as 2D
        # to include the 1-sigma range)
        phys.chainFileToStatsFile(
            chainfname,
            physfname,
            burn=0, randsamples=1000
        )

    print('Done', name)

if __name__ == '__main__':
    fit()
