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

# Convenience routines to help the user

import numpy as np

import astropy.wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord

from . import data

def imageLoad(img_fname, exp_fname,
              rmf='rmf.fits', arf='arf.fits',
              emin_keV=0.5, emax_keV=2.0,
              exp_novig_fname=None,
              origin=None, pix_origin=None,
              mask_fname=None, psf=None,
              band_name=None):
    """A helper to construct an Image from input fits images

    img_fname: input fits image filename
    exp_fname: input exposure fits image filename
    rmf: response filename
    arf: arf filename
    emin_keV: minimum energy (keV)
    emax_keV: maximum energy (keV)
    
    Optional:
     exp_novig_fname: unvignetted exposure filename
     origin: origin to measure coordinates from (ra_deg, dec_deg) or SkyCoord.
       By default uses FITS reference coordinate.
     pix_origin: override above with origin (y,x) in pixels
     mask_fname: name of filename to get mask image
     band_name: name for band (default X.XX_Y.YY with emin/emax)

    Exposure maps loaded are given names 'expmap' and optionally 'expmap_novig'
    """

    if not band_name:
        band_name = '%g_%g' % (emin_keV, emax_keV)

    imgf = fits.open(img_fname, 'readonly')
    hdu = imgf[0]
    hdr = hdu.header
    img = hdu.data
    wcs = astropy.wcs.WCS(hdu)

    # get pixel size in arcsec and check pixels are square
    assert hdr['CUNIT1'] == 'deg'
    assert np.isclose(abs(hdr['CDELT1']), abs(hdr['CDELT2']))
    pixsize_as = abs(hdr['CDELT1'])*3600

    # get image origin (in pixels or coordinate)
    if pix_origin is not None:
        if origin is None:
            # use CRPIX coordinates
            pix_origin = (hdr['CRPIX2']-1, hdr['CRPIX1']-1)
        else:
            if not isinstance(origin, SkyCoord):
                orgin = SkyCoord(origin[0], origin[1], unit='deg')
            x, y = astropy.wcs.utils.skycoord_to_pixel(origin, wcs, 0)
            pix_origin = (y[0], x[0])

    imgf.close()

    # load exposure map
    expf = fits.open(exp_fname, 'readonly')
    expimg = expf[0].data
    expf.close()
    expmaps = {'expmap': expimg}

    # optional non-vignetted exposure
    if exp_novig_fname:
        expf = fits.open(exp_novig_fname, 'readonly')
        expimg = expf[0].data
        expf.close()
        expmaps['expmap_novig'] = expmap

    # load mask image
    if mask_fname:
        maskf = fits.open(mask_fname, 'readonly')
        mask = maskf[0].data
        mask = mask & (expimg>0)
        maskf.close()
        assert mask.shape == img.shape
    else:
        mask = expimg>0

    assert expimg.shape == img.shape

    img = data.Image(
        band_name,
        img,
        rmf=rmf,
        arf=arf,
        pixsize_as=pixsize_as,
        expmaps=expmaps,
        mask=mask,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
        origin=pix_origin,
        psf=psf,
        wcs=wcs,
    )

    return img

def PSFLoad(filename, hdu):
    """Load a PSF image from filename given index/name of HDU.

    PSF should have CRPIX1/2 as origin and CDELT1 as pixel size (arcsec)
    """

    psff = fits.open(filename, 'readonly')

    # construct PSF object using image from appropriate HDU
    hdu = psff[hdu]
    psfimg = hdu.data
    #  this is the centre of the PSF (y,x)
    psforig = (hdu.header['CRPIX2']-1, hdu.header['CRPIX1']-1)

    # pixel size in arcsec
    assert hdu.header['CUNIT1'] == 'arcsec'
    psfpixsize_as = abs(hdu.header['CDELT1'])
    psf = data.PSF(psfimg, psfpixsize_as, origin=psforig)

    return psf
