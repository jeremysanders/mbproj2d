// Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#ifndef PROJECT_CC_HH
#define PROJECT_CC_HH

void project(const float rbin, const int numbins,
             const float* emiss, float* sb);

void add_sb_prof(const float rbin, const int nbins, const float *sb,
		 const float xc, const float yc,
		 const int xw, const int yw,
		 float* img);

void add_sb_prof_e(const float rbin, const int nbins, const float *sb,
		   const float xc, const float yc,
		   const float e, const float theta,
		   const int xw, const int yw,
		   float* img);

double logLikelihood(const int nelem, const float* data, const float* model);
double logLikelihoodMasked(int nelem, const float* data, const float* model,
                           const int* mask);

void resamplePSF(int psf_nx, int psf_ny,
                 double psf_pixsize,
                 double psf_ox, double psf_oy,
                 const float *psf,
                 int img_nx, int img_ny,
                 double img_pixsize,
                 float *img);

void clipMin(float minval, int ny, int nx, float* arr);
void clipMax(float maxval, int ny, int nx, float* arr);

#endif
