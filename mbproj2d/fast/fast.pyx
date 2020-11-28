# distutils: language = c++
#cython: language_level=3

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
    float logLikelihoodMasked(const int nelem, const float* data, const float* model, const int* mask)
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

cdef extern from "binimg_cc.hh":
    void accreteBinImage_cc(int xw, int yw, const float *inimg, const int *mask,
                            float thresh, int *binimg)
    void buildVoronoiMap_cc(int xw, int yw, const float *inimg,
                            int *binimg)

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
    return logLikelihood(nelem, &data_view[0,0], &model_view[0,0])

@cython.boundscheck(False)
@cython.wraparound(False)
def calcPoissonLogLikelihoodMasked(float[:,::1] data, float[:,::1] model, int[:,::1] mask, precise=False):

    assert data.shape[0] == model.shape[0]
    assert mask.shape[0] == model.shape[0]
    assert data.shape[1] == model.shape[1]
    assert mask.shape[1] == model.shape[1]

    cdef int nelem
    nelem = model.shape[0]*model.shape[1]
    return logLikelihoodMasked(nelem, &data[0,0], &model[0,0], &mask[0,0])

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

@cython.boundscheck(False)
@cython.wraparound(False)
def accreteBinImage(float[:,::1] inimg, int[:,::1] mask, double thresh):
    assert inimg.shape[0] == mask.shape[0]
    assert inimg.shape[1] == mask.shape[1]

    cdef int[:,::1] out_view

    out = np.empty((inimg.shape[0], inimg.shape[1]), dtype=np.int32)
    out_view = out

    accreteBinImage_cc(inimg.shape[1], inimg.shape[0], &inimg[0,0],
                       &mask[0,0], thresh, &out_view[0,0])

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def buildVoronoiMap(float[:,::1] inimg, int[:,::1] binmap):
    assert inimg.shape[0] == binmap.shape[0]
    assert inimg.shape[1] == binmap.shape[1]

    buildVoronoiMap_cc(inimg.shape[0], inimg.shape[1], &inimg[0,0],
                       &binmap[0,0])
