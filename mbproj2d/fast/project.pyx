# distutils: language = c++
#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

cdef extern from "project_cc.hh":
    void project(float rbin, int numbins,
                 const float* emiss, float* sb)
    void add_sb_prof(float rbin, int nbins, const float *sb,
    		     float xc, float yc,
		     int xw, int yw, float *img)
    double logLikelihood(const int nelem, const float* data, const float* model)
    float logLikelihoodAVX(const int nelem, const float* data, const float* model)
    float logLikelihoodAVXMasked(const int nelem, const float* data, const float* model, const int* mask)
    void resamplePSF(int psf_nx, int psf_ny,
                     float psf_pixsize,
                     float psf_ox, float psf_oy,
                     const float *psf,
                     int oversample,
                     int img_nx, int img_ny,
                     float img_pixsize,
                     float *img)
    void clipMin(float minval, int ny, int nx, float* arr)
    void clipMax(float maxval, int ny, int nx, float* arr)

@cython.boundscheck(False)
@cython.wraparound(False)
def projectEmissivity(float rbin, np.ndarray emiss):
    assert emiss.dtype == np.float32

    sb = np.empty(emiss.shape[0], dtype=np.float32)

    cdef float[::1] emiss_view = emiss
    cdef float[::1] sb_view = sb
    cdef int numbins

    numbins = emiss_view.shape[0]
    project(rbin, numbins, &emiss_view[0], &sb_view[0])

    return sb

# note: final element in SB profile should be zero
@cython.boundscheck(False)
@cython.wraparound(False)
def addSBToImg(float rbin, float[::1] sb, float xc, float yc,
               float[:,::1] img):

    cdef int numbins, xw, yw

    numbins = sb.shape[0]
    yw = img.shape[0]
    xw = img.shape[1]

    add_sb_prof(
        rbin, numbins, &sb[0], xc, yc, xw, yw,
        &img[0,0]
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def calcPoissonLogLikelihood(np.ndarray data, np.ndarray model):
    assert data.dtype == np.float32
    assert model.dtype == np.float32
    assert data.shape[0] == model.shape[0]
    assert data.shape[1] == model.shape[1]

    cdef float[:,::1] data_view = data
    cdef float[:,::1] model_view = model
    cdef int nelem

    nelem = data.shape[0]*data.shape[1]
    #return logLikelihood(nelem, &data_view[0,0], &model_view[0,0])
    return logLikelihoodAVX(nelem, &data_view[0,0], &model_view[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def calcPoissonLogLikelihoodMasked(float[:,::1] data, float[:,::1] model, int[:,::1] mask):

    assert data.shape[0] == model.shape[0]
    assert mask.shape[0] == model.shape[0]
    assert data.shape[1] == model.shape[1]
    assert mask.shape[1] == model.shape[1]

    cdef int nelem
    nelem = model.shape[0]*model.shape[1]
    return logLikelihoodAVXMasked(nelem, &data[0,0], &model[0,0], &mask[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def resamplePSFImage(float[:,::1] psfimg,
                     float[:,::1] outimg,
                     float psf_pixsize=1,
                     float img_pixsize=1,
                     float psf_ox=0,
                     float psf_oy=0,
                     int oversample=16):

    resamplePSF(psfimg.shape[1], psfimg.shape[0],
                psf_pixsize,
                psf_ox, psf_oy,
                &psfimg[0,0],
                oversample,
                outimg.shape[1], outimg.shape[0],
                img_pixsize,
                &outimg[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def clip2DMin(float[:,::1] img, float val):
    clipMin(val, img.shape[0], img.shape[1], &img[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def clip2DMax(float[:,::1] img, float val):
    clipMax(val, img.shape[0], img.shape[1], &img[0,0])
