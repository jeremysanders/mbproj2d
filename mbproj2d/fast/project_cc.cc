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

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

using std::printf;
using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::log;

template<class T> inline T sqr(T a)
{
  return a*a;
}

// calculate a**2 - b**2
template<class T> inline T delta_sqr(T a, T b)
{
  return (a+b)*(a-b);
};

// calculate a**3 - b**3
template<class T> inline T delta_cube(T a, T b)
{
  return (a-b)*(a*a+a*b+b*b);
};

// square root of difference of squares
template<class T> inline T sqrt_delta_sqr(T a, T b)
{
  return sqrt(delta_sqr(a,b));
}

// get an integer from float, rounding away from 0
inline int round_away_from_zero(float x)
{
  return int((x<0) ? floor(x) : ceil(x));
}


// this is several times slower than doing the matrix multiplication
// 229 us vs 49 us for 300 element profile
void project(const float rbin, const int numbins,
             const float* emiss, float* sb)
{
  // separate out the volume here, and deal with steps of 1 to improve
  // adding accuracy
  const float volfactor = 4./3.*M_PI * rbin*rbin*rbin;

  float yf0 = 0;
  for(int yi=0; yi<numbins; ++yi)
    {
      const float yf1 = yf0 + 1;

      float rf0 = yf0;
      float tot = 0;
      for(int ri=yi; ri<numbins; ++ri)
	{
          const float rf1 = rf0 + 1;

          const float p1 = (ri   <= yi+1) ? 0 : sqrt_delta_sqr(rf0, yf1);
          const float p2 = (ri   <= yi  ) ? 0 : sqrt_delta_sqr(rf0, yf0);
          const float p3 = (ri+1 <= yi+1) ? 0 : sqrt_delta_sqr(rf1, yf1);
          const float p4 = (ri+1 <= yi  ) ? 0 : sqrt_delta_sqr(rf1, yf0);
	  const float vol = delta_cube(p1, p2) + delta_cube(p4, p3);

	  tot += vol * emiss[ri];
          rf0 = rf1;
	}
      sb[yi] = volfactor * tot;

      yf0 = yf1;
    }
}

void add_sb_prof(const float rbin, const int nbins, const float *sb,
		 const float xc, const float yc,
		 const int xw, const int yw,
		 float *img)
{
  // copy to ensure outer bin is 0
  std::vector<float> cpy_sb(sb, sb+nbins);
  cpy_sb.push_back(0);

  const int x1 = max(int(xc-rbin*nbins), 0);
  const int x2 = min(int(xc+rbin*nbins), xw-1);
  const int y1 = max(int(yc-rbin*nbins), 0);
  const int y2 = min(int(yc+rbin*nbins), yw-1);

  const float invrbin = 1/rbin;

  for(int y=y1; y<=y2; ++y)
    for(int x=x1; x<=x2; ++x)
      {
	const float r = sqrt(sqr(x-xc) + sqr(y-yc)) * invrbin;

	const int i0 = min(int(r), nbins);
	const int i1 = min(i0+1, nbins);
	const float w1 = r-int(r);
	const float w0 = 1-w1;

	const float val = cpy_sb[i0]*w0 + cpy_sb[i1]*w1;

	img[y*xw+x] += val;
      }
}

// calculate a Poisson log likelihood (direct)
double logLikelihood(const int nelem, const float* data, const float* model)
{
  double sumdm = 0;
  double summ = 0;
  for(int i=0; i<nelem; ++i)
    {
      if(data[i] != 0)
        sumdm += data[i]*log(model[i]);
      summ -= model[i];
    }
  return sumdm + summ;
}

double logLikelihoodMasked(int nelem, const float* data, const float* model, const int* mask)
{
  // keeps track of the two components before adding them at the end
  double sum1 = 0;
  double sum2 = 0;
  for(int i=0; i<nelem; ++i)
    {
      if(mask[i])
        {
          // it's a lot faster not to calculate a log, so avoid doing
          // so if it's multiplied by zero
          if(data[i] != 0)
            sum1 += double(data[i]*log(model[i]));
          sum2 -= double(model[i]);
        }
    }
  return sum1 + sum2;
}

// resample a PSF image between two different image resolutions
void resamplePSF(int psf_nx, int psf_ny,
                 float psf_pixsize,
                 float psf_ox, float psf_oy,
                 const float *psf,
                 int oversample,
                 int img_nx, int img_ny,
                 float img_pixsize,
                 float *img)
{
  oversample = max(oversample, 1);
  const float inv_over = 1.0f/oversample;
  const float pix_ratio = img_pixsize / psf_pixsize;

  double img_tot = 0;
  for(int y=0; y<img_ny; ++y)
    for(int x=0; x<img_nx; ++x)
      {
        // want to put PSF at corners of output image
        const int wrapy = y<img_ny/2 ? y : y-img_ny;
        const int wrapx = x<img_nx/2 ? x : x-img_nx;

        float tot = 0;
        int num = 0;
        for(int sy=0; sy<oversample; ++sy)
          for(int sx=0; sx<oversample; ++sx)
            {
              const int psfy = int((wrapy+sy*inv_over)*pix_ratio + psf_oy);
              const int psfx = int((wrapx+sx*inv_over)*pix_ratio + psf_ox);
              if(psfy >=0 && psfy < psf_ny && psfx >= 0 && psfx < psf_nx)
                {
                  tot += psf[ psfy*psf_nx + psfx ];
                  ++num;
                }
            }
        img[ y*img_nx + x ] = num>0 ? tot/num : 0;
        img_tot += img[ y*img_nx + x ];
      }

  // normalise image
  for(int y=0; y<img_ny; ++y)
    for(int x=0; x<img_nx; ++x)
      img[ y*img_nx + x ] /= img_tot;
}

void clipMin(float minval, int ny, int nx, float* arr)
{
  for(int i=0; i<(nx*ny); ++i)
    arr[i] = std::min(arr[i], minval);
}

void clipMax(float maxval, int ny, int nx, float* arr)
{
  for(int i=0; i<(nx*ny); ++i)
    arr[i] = std::max(arr[i], maxval);
}
