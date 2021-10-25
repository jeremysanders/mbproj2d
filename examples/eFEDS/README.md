## eFEDS example

This is an example fitting clusters in the public eFEDS data, for a subset of the eFEDS field.

Please see
* Brunner et al.: https://arxiv.org/abs/2110.09544
* Liu et al.: https://arxiv.org/abs/2106.14518
* eROSITA data and calibration: https://erosita.mpe.mpg.de/edr/

It fits two clusters simultaneously with isothermal models.

Notes:
1. The energy bands used are 0.3-0.6, 0.6-1.1, 1.1-1.6, 1.6-2.2, 2.2-3.5, 3.5-5.0 and 5.0-7.0 keV.
2. PSF modelling is included using, the survey-averaged eROSITA PSF.
3. The exposure maps are in seconds, and have been rescaled to have the appropriate exposure for the response given the vignetting of the telescope.
4. The response matrix was created using srctool, images were made using evtool, and exposure maps made by expmap.
5. A very basic background model is used - it is assumed that the background is flat in each energy band.
6. The MCMC chain output has not been tested for convergence.
7. The mask file masks out point sources. These are not currently included in the model.
