#!/usr/bin/env python3

import pickle
import os

import numpy as N

import astropy.cosmology
import astropy.units as u
from astropy.io import fits

import mbproj2d as mb

cosmo = astropy.cosmology.FlatLambdaCDM(70., 0.3)

bands_keV = [
    [0.5, 0.8],
    [0.8, 1.2],
    [1.2, 2.0],
    [2.0, 3.0],
    [3.0, 5.0],
    [5.0, 7.0],
]

# responses to use
rmf = 'test.rmf'
arf = 'test.arf'

# exposure time to simulate
exposure_s = 60000

# image size
imgsize = [512, 512]
centre = [imgsize[0]/2, imgsize[1]/2]
pixsize_arcsec = 1.0

# output filenames
model_fname = 'hydrostatic_model.fits'
data_fname = 'hydrostatic_real.fits'
pars_save_fname = 'hydrostatic.pars'
fit_save_fname = 'hydrostatic.fit'
chain_fname = 'hydrostatic.chain.hdf5'
phys_fname = 'hydrostatic.phys.hdf5'

# redshift of system
z = 0.3
# absorbing photoelect. column density
NH_1022pcm2 = 0.02

def calc_delta_c(c):
    """Get characteristic overdensity for concentration."""
    return (200/3.) * c**3 / (N.log(1+c) - c/(1+c))

def calc_r_s(R200, c):
    """Get scale radius given R200 and concentration."""
    return R200 / c

def calc_rho_crit(z):
    """Get critical density at redshift in g/cm3."""
    return cosmo.critical_density(z).to_value(u.g/u.cm**3)

def calc_nfw_rho(r, R200, c, z):
    """Get NFW density profile at radius."""
    delta_c = calc_delta_c(c)
    rho_crit = calc_rho_crit(z)
    r_s = calc_r_s(R200, c)

    x = r/r_s
    rho = delta_c * rho_crit / (x * (1+x)**2)
    return rho

def calc_beta_core(r, ne0, r_c, beta, alpha):
    """Get electron density profile.

    This is a beta model with a core."""

    ne2 =  ne0**2 * (r/r_c)**(-alpha) / ((1+r**2/r_c**2)**(3*beta-0.5*alpha))
    return N.sqrt(ne2)

def do_simulation():
    # input parameters
    R200_cm = 1.2 * mb.Mpc_cm # R200
    conc = 5                  # concentration
    beta = 2./3.              # beta slope
    ne0_pcm3 = 0.005          # inner density
    alpha = 0.2               # inner slope
    rc_cm = 300. * mb.kpc_cm  # core radius
    P0_ergpcm3 = 1e-14        # outer pressure
    Z_solar = 0.3             # metallicity (solar)

    # outer radius
    max_r_cm = 4.0 * mb.Mpc_cm
    pixsize_cm = (
        1/cosmo.arcsec_per_kpc_proper(z)*u.arcsec*pixsize_arcsec).to_value(u.cm)
    numpix = int(max_r_cm / pixsize_cm) + 1

    # radii of shells evaluating with
    rshell_pix = N.arange(numpix+1)
    rshell_cm = rshell_pix * pixsize_cm
    r_cm = 0.5*(rshell_cm[1:]+rshell_cm[:-1])

    # DM NFW density profile
    nfw_rho_gpcm3 = calc_nfw_rho(r_cm, R200_cm, conc, z)

    # this is the calculated density profile
    ne_pcm3 = calc_beta_core(r_cm, ne0_pcm3, rc_cm, beta, alpha)
    gas_rho_gpcm3 = mb.mu_e * mb.mu_g * ne_pcm3

    # total density
    tot_rho_gpcm3 = nfw_rho_gpcm3 + gas_rho_gpcm3

    # calc gravitational acceleration
    Mshell = (4/3.)*N.pi*(rshell_cm[1:]**3-rshell_cm[:-1]**3) * tot_rho_gpcm3
    print(N.sum(Mshell))

    g_cmps2 = mb.G_cgs * N.cumsum(Mshell) / r_cm**2

    deltaP_ergpcm3 = gas_rho_gpcm3 * g_cmps2 * (rshell_cm[1:]-rshell_cm[:-1])
    P_ergpcm3 = N.cumsum(deltaP_ergpcm3[::-1])[::-1] + P0_ergpcm3

    T_keV = P_ergpcm3 / (mb.P_keV_to_erg * ne_pcm3)

    # project emissivity to SB
    proj_matrix = mb.projectionVolumeMatrix(rshell_pix) / (
        N.pi*(rshell_pix[1:]**2-rshell_pix[:-1]**2))[:,N.newaxis]

    xs = mb.XSpecHelper()
    xs.setAbund('lodd')
    imgs = N.zeros((len(bands_keV), imgsize[0], imgsize[1]), dtype=N.float32)
    for i, band in enumerate(bands_keV):
        xs.changeResponse(rmf, arf, band[0], band[1])

        # calculate norm per cubic pix
        norms = (
            1e-14/(4*N.pi*(cosmo.angular_diameter_distance(z).to_value(u.cm)*(1+z))**2) *
            ne_pcm3**2 * (1+1/mb.ne_nH) * pixsize_cm**3)

        emiss = []
        for T, norm in zip(T_keV, norms):
            xs.setApec(NH_1022pcm2, T, Z_solar, z, norm)
            emiss.append(xs.getRate())

        sb = proj_matrix.dot(emiss).astype(N.float32) * exposure_s
        mb.addSBToImg(1, sb, centre[0], centre[1], imgs[i])

    fout = fits.HDUList([fits.PrimaryHDU(imgs)])
    fout.writeto(model_fname, overwrite=True)

    real = N.random.poisson(lam=imgs).astype(N.int32)
    fout = fits.HDUList([fits.PrimaryHDU(real)])
    fout.writeto(data_fname, overwrite=True)

def fit_simulation():

    fin = fits.open(data_fname, 'readonly')
    data = N.array(fin[0].data)
    fin.close()

    images = []
    for i, band in enumerate(bands_keV):
        img = data[i, :, :]
        expimg = N.full(img.shape, exposure_s)
        
        image = mb.Image(
            '%0.1g_%0.1g' % tuple(band),
            img,
            rmf=rmf,
            arf=arf,
            pixsize_as=pixsize_arcsec,
            expmaps={'object': expimg},
            emin_keV=band[0], emax_keV=band[1],
            origin=(centre[0], centre[1]),
            psf=None,
        )
        images.append(image)

    pars = mb.Pars()
    cosmo = mb.Cosmology(z)

    ne_prof = mb.ProfileVikhDensity('ne', pars, mode='betacore')
    Z_prof = mb.ProfileFlat('Z', pars, defval=0.3)
    pars['Z'].frozen = True

    mass_prof = mb.ProfileMassNFW('nfw', cosmo, pars)

    cluster = mb.ClusterHydro(
        'cluster',
        pars,
        images,
        cosmo=cosmo,
        NH_1022pcm2=NH_1022pcm2,
        ne_prof=ne_prof,
        mass_prof=mass_prof,
        Z_prof=Z_prof,
        cx=2,
        cy=2,
    )

    totmod = mb.TotalModel(
        pars,
        images,
        src_expmap='object',
        src_models=[cluster],
        back_models=[],
    )

    if os.path.exists(pars_save_fname):
        with open(pars_save_fname, 'rb') as f:
            pars = pickle.load(f)

    fit = mb.Fit(images, totmod, pars)
    pars.write()
    fit.run()
    pars.write()

    pars.save(pars_save_fname)
    fit.save(fit_save_fname)

    # Do MCMC
    mcmc = mb.MCMC(fit, processes=10, nwalkers=100)
    mcmc.burnIn(1000)
    mcmc.run(1000)
    mcmc.save(chain_fname)

    # Then compute physical profiles
    phys = mb.Phys(pars, cluster)
    phys.chainFileToStatsFile(chain_fname, phys_fname, burn=0, randsamples=1000)

if __name__ == '__main__':
    do_simulation()
    fit_simulation()
