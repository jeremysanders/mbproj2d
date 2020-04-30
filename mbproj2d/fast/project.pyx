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
		     int xw, int yw, const float* expmap,
                     float *img)
    double logLikelihood(const int nelem, const float* data, const float* model)
    float logLikelihoodAVX(const int nelem, const float* data, const float* model)

@cython.boundscheck(False)
@cython.wraparound(False)
def projectEmissivity(float rbin, np.ndarray emiss):
    assert emiss.dtype == np.float32

    sb = np.empty(emiss.shape[0], dtype=np.float32)

    cdef float[::1] emiss_view = emiss
    cdef float[::1] sb_view = sb
    cdef int numbins

    numbins = emiss.shape[0]
    project(rbin, numbins, &emiss_view[0], &sb_view[0])

    return sb

# note: final element in SB profile should be zero
@cython.boundscheck(False)
@cython.wraparound(False)
def addSBToImg(float rbin, np.ndarray sb, float xc, float yc,
               np.ndarray expmap, np.ndarray img):
    assert sb.dtype == np.float32
    assert img.dtype == np.float32
    assert expmap.dtype == np.float32

    cdef float[::1] sb_view = sb
    cdef float[:,::1] img_view = img
    cdef float[:,::1] expmap_view = expmap
    cdef int numbins, xw, yw

    numbins = sb.shape[0]
    yw = img.shape[0]
    xw = img.shape[1]

    assert xw == expmap.shape[0]
    assert yw == expmap.shape[1]

    add_sb_prof(
        rbin, numbins, &sb_view[0], xc, yc, xw, yw,
        &expmap_view[0,0],
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
