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
def addSBToImg(float rbin, np.ndarray sb, float xc, float yc,
               np.ndarray img):
    assert sb.dtype == np.float32
    assert img.dtype == np.float32

    cdef float[::1] sb_view = sb
    cdef float[:,::1] img_view = img
    cdef int numbins, xw, yw

    numbins = sb_view.shape[0]
    yw = img_view.shape[0]
    xw = img_view.shape[1]

    add_sb_prof(
        rbin, numbins, &sb_view[0], xc, yc, xw, yw,
        &img_view[0,0]
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
def calcPoissonLogLikelihoodMasked(np.ndarray data, np.ndarray model, np.ndarray mask):
    assert data.dtype == np.float32
    assert model.dtype == np.float32
    assert mask.dtype == np.int32
    assert data.shape[0] == model.shape[0]
    assert mask.shape[0] == model.shape[0]
    assert data.shape[1] == model.shape[1]
    assert mask.shape[1] == model.shape[1]

    cdef float[:,::1] data_view = data
    cdef float[:,::1] model_view = model
    cdef int[:,::1] mask_view = mask
    cdef int nelem

    nelem = model.shape[0]*model.shape[1]
    return logLikelihoodAVXMasked(nelem, &data_view[0,0], &model_view[0,0], &mask_view[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def resamplePSFImage(np.ndarray psfimg,
                     np.ndarray outimg,
                     float psf_pixsize=1,
                     float img_pixsize=1,
                     float psf_ox=0,
                     float psf_oy=0,
                     int oversample=16):
    assert psfimg.dtype == np.float32
    assert outimg.dtype == np.float32

    cdef float[:,::1] psf_view = psfimg
    cdef float[:,::1] out_view = outimg

    resamplePSF(psf_view.shape[1], psf_view.shape[0],
                psf_pixsize,
                psf_ox, psf_oy,
                &psf_view[0,0],
                oversample,
                out_view.shape[1], out_view.shape[0],
                img_pixsize,
                &out_view[0,0])
