#!/usr/bin/env python3

# This is a test program to simulate a simple cluster model and to
# refit it with mbproj2d

# We make a big cube to calculate the counts per unit volume, then
# integrate along a line of sight to compute a surface brightness map

import os
import sys
sys.path.append('../')

import mbproj2d as mb

import math
import h5py
import numpy as N
from astropy.io import fits

kpc_cm = 3.0856776e21
Mpc_cm = 3.0856776e24

redshift = 0.3
as_kpc = 4.454
DL_cm = 1552.7 * Mpc_cm
DA_cm = 918.7 * Mpc_cm
NH_1022pcm2 = 0.01

# rates for emission measure = 1 for model.xcm with erosita onaxis rmf,arf
# model  TBabs*apec
#            0.01      0.001          0          0     100000      1e+06
#               5       0.01      0.008      0.008         64         64
#             0.3     -0.001          0          0          5          5
#             0.3      -0.01     -0.999     -0.999         10         10
#               1       0.01          0          0      1e+20      1e+24

bands = [(0.5,1.0), (1.0,2.0), (2.0,4.0), (4.0,7.0)]
conv_rates = [216.320, 217.758, 45.9045, 9.06225]

pixsize_as = 2.
maxrad_kpc = 3500.
expos_s = 20e3
ne_nH = 1.2

beta = 0.66
rc_kpc = 200.
ne0 = 0.01

bkg_cts = 1e-2

pix_kpc = pixsize_as * as_kpc

rmf = 'onaxis_020_RMF_00001.fits'
arf = 'onaxis_020_ARF_00001.fits'

pars_fname = '3d_model_pars.pickle'
chain_fname = '3d_model_chain.hdf5'
sb_fname = '3d_model_sb.hdf5'
phys_fname = '3d_model_phys.hdf5'

fake_fname = '3d_model_fake.fits'
mod_fname = '3d_model_model.fits'

def getNorm(r_kpc):
    """For a cube in the 2D grid, calculate a norm."""
    ne = ne0 * (1 + (r_kpc * (1/rc_kpc))**2) ** (-3*beta/2)
    nenH = ne**2 * (1/ne_nH)
    dV_cm3 = (pix_kpc*kpc_cm)**3
    factor = (1e-14 / (4*math.pi*(DA_cm*(1+redshift))**2)) * dV_cm3

    norm = nenH * factor
    return norm

def getModelImages():
    print('Making input models')
    if os.path.exists(mod_fname):
        print("Already exists")
        return

    # number of pixels and centre
    npix = int(maxrad_kpc / pix_kpc)*2
    c = npix/2

    # radius in kpc of each pixel in 3D grid
    r_kpc = N.fromfunction(
        lambda z,y,x: N.sqrt((z-c)**2+(y-c)**2+(x-c)**2)*pix_kpc,
        (npix,npix,npix), dtype=N.float32)

    # compute norm in each pixel
    images = []
    norm = getNorm(r_kpc)

    # use same cut off radius as cluster model default
    norm[r_kpc > 3000.] = 0

    # now convert to cts using the rates above
    for band, rate in zip(bands, conv_rates):
        ctscube = norm*(rate*expos_s)

        ctsimg = ctscube.sum(axis=0)
        images.append(ctsimg)

    # write some model images
    ff = fits.HDUList([fits.PrimaryHDU(N.array(images))])
    ff.writeto(mod_fname, overwrite=True)

def fakeImages():
    print('Making input images')
    if os.path.exists(fake_fname):
        print("Already exists")
        return

    with fits.open(mod_fname) as fin:
        modimgs = fin[0].data + 0

    # fixed random number generator
    rs = N.random.RandomState(42)

    fakeimgs = []
    for img in modimgs:
        img = img + bkg_cts

        fake = rs.poisson(img).astype(N.float32)
        fakeimgs.append(fake)

    ff = fits.HDUList([fits.PrimaryHDU(N.array(fakeimgs).astype(N.int16))])
    ff.writeto(fake_fname, overwrite=True)

def runAnalysis():
    print('Running analysis')

    with fits.open(fake_fname) as fin:
        imgarrs = fin[0].data + 0

    images = []
    for band, arr in zip(bands, imgarrs):
        # flat exposure map
        expmap = expos_s + arr*0

        img = mb.Image(
            '%g_%g' % band,
            arr,
            rmf=rmf, arf=arf,
            pixsize_as=pixsize_as,
            expmaps={'object': expmap},
            emin_keV=band[0], emax_keV=band[1],
            psf=None,
            )
        images.append(img)

    pars = mb.Pars()
    cosmo = mb.Cosmology(redshift)
    ne_prof = mb.ProfileBeta('ne', pars)

    T_prof = mb.ProfileFlat('T', pars, defval=3.)
    Z_prof = mb.ProfileFlat('Z', pars, defval=0.3)
    pars['Z'].frozen = True

    cluster = mb.ClusterNonHydro(
        'cluster',
        pars,
        images,
        cosmo=cosmo,
        NH_1022pcm2=NH_1022pcm2,
        ne_prof=ne_prof,
        T_prof=T_prof,
        Z_prof=Z_prof,
        cx=pixsize_as*imgarrs[0].shape[1]/2*1.05,
        cy=pixsize_as*imgarrs[0].shape[0]/2/1.05,
    )

    backmod = mb.BackModelFlat(
        'back',
        pars,
        images,
        log=True
    )

    # change the background rates
    pars['back_0.5_1'].val = -3
    pars['back_1_2'].val = -3
    pars['back_2_4'].val = -3
    pars['back_4_7'].val = -3

    totmod = mb.TotalModel(
        pars,
        images,
        src_expmap='object',
        src_models=[cluster],
        back_models=[backmod],
    )

    pars.write()

    fit = mb.Fit(images, totmod, pars)

    if os.path.exists(pars_fname):
        print('loading', pars_fname)
        pars.load(pars_fname)
    else:
        # find best fitting parameters
        pars.write()
        fit.run(maxloops=1)
        pars.write()
        fit.run()
        # save parameters
        pars.save(pars_fname)
    pars.write()

    if not os.path.exists(chain_fname):
        print('Running MCMC')
        mcmc = mb.MCMC(fit, processes=4)
        mcmc.burnIn(2000)
        mcmc.run(2000)
        mcmc.save(chain_fname)

    if not os.path.exists(sb_fname):
        print('Calculating SB')
        sbmaps = mb.SBMaps(
            pars, totmod, images,
            prof_origin=(pars['cluster_cy'].v, pars['cluster_cx'].v),
            make_cvtmaps=False,
        )
        output = sbmaps.calcStats(
            mb.loadChainFromFile(chain_fname, pars, randsamples=1000),
            h5fname=sb_fname,
        )

    if not os.path.exists(phys_fname):
        print('Calculating physics')
        phys = mb.Phys(pars, cluster, rate_rmf=rmf, rate_arf=arf, rate_bands=bands, rmax_kpc=3000, rsteps=1024)
        phys.chainFileToStatsFile(
            chain_fname, phys_fname, burn=0, randsamples=1000)

def main():
    getModelImages()
    fakeImages()
    runAnalysis()

if __name__ == '__main__':
    main()
