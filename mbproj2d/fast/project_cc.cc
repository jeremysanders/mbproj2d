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
using std::round;
using std::min;
using std::max;
using std::log;
using std::sin;
using std::cos;
using std::atan2;
using std::pow;

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

namespace {
  // do linear interpolation of r into sb array
  inline float linear_interpolate(const float r, const std::vector<float>& sb, const int nbins)
  {
    // interpolate from the centre of the pixel bins - shift inwards by 0.5 to do this
    // this is necessary for consistency on rebinning
    const float rc = max(r-0.5f, 0.f);

    const int ri = int(rc);
    const int i0 = min(ri, nbins);
    const int i1 = min(i0+1, nbins);
    const float w1 = rc-ri;
    const float w0 = 1-w1;
    const float val = sb[i0]*w0 + sb[i1]*w1;
    return val;
  }
}

// paint surface brightness to image
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
        const float val = linear_interpolate(r, cpy_sb, nbins);
        img[y*xw+x] += val;
      }
}

// elliptical version of painting a profile
void add_sb_prof_e(const float rbin, const int nbins, const float *sb,
                   const float xc, const float yc,
                   const float e, const double theta,
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
  // const float inve = 1/e;
  const float sqe = sqrt(e);
  const float sqinve = sqrt(1/e);

  const float s = float(sin(theta*(M_PI/180)));
  const float c = float(cos(theta*(M_PI/180)));

  for(int y=y1; y<=y2; ++y)
    for(int x=x1; x<=x2; ++x)
      {
        const float dx = x-xc;
        const float dy = y-yc;
        const float rx = (c*dx - s*dy)*sqe;
        const float ry = (s*dx + c*dy)*sqinve;

        // interpolate from centre of pixel bins
        const float r = sqrt(sqr(rx) + sqr(ry)) * invrbin;
        const float val = linear_interpolate(r, cpy_sb, nbins);
        img[y*xw+x] += val;
      }
}

// slosh version of painting a profile
void add_sb_prof_slosh(const float rbin, const int nbins, const float *sb,
                       const float xc, const float yc,
                       const float slosh, const double theta0,
                       const int xw, const int yw,
                       float *img)
{
  // copy to ensure outer bin is 0
  std::vector<float> cpy_sb(sb, sb+nbins);
  cpy_sb.push_back(0);

  // expand box considered to avoid edge
  const float maxr = rbin*nbins / (1-slosh);
  const int x1 = max(int(xc-maxr), 0);
  const int x2 = min(int(xc+maxr), xw-1);
  const int y1 = max(int(yc-maxr), 0);
  const int y2 = min(int(yc+maxr), yw-1);

  const float invrbin = 1/rbin;
  const float s = float(sin(theta0*(M_PI/180)));
  const float c = float(cos(theta0*(M_PI/180)));

  // this is the scaling factor for scaling the flux of the model for a given slosh
  // to compensate for the area scaling

  // Consider a circle that is sloshed, and calculate the area relative to a circle
  // We want to integrate sqrt(x**2+y**2) + slosh*x < 1, and divide by this (relative to slosh=0)

  // From Mathematica (here s is slosh):
  //   Pi/Integrate[
  //     Integrate[
  //      1, {x, (s - Sqrt[1 - y^2 + s^2 y^2])/(-1 + s^2), (
  //       s + Sqrt[1 - y^2 + s^2 y^2])/(-1 + s^2)}], {y, -(1/Sqrt[1 - s^2]),
  //       1/Sqrt[1 - s^2]}]
  // Equals (1-s**2)**(3/2)
  const float sloshscale = pow(1-sqr(slosh), 1.5f);

  // now scale sb vector by this, so we don't need to compute for each pixel
  for(size_t i=0; i != cpy_sb.size(); ++i)
    cpy_sb[i] *= sloshscale;

  for(int y=y1; y<=y2; ++y)
    for(int x=x1; x<=x2; ++x)
      {
        const float dx = x-xc;
        const float dy = y-yc;
        const float rold = sqrt(sqr(dx)+sqr(dy)) * invrbin;

        // rotate so theta0 is along axis
        const float rx = dx*c - dy*s;

        // this is equivalent to
        //  rslosh = rold * (slosh*cos(atan2(dy,dx)+theta0_rad) + 1)
        const float rslosh = rold + slosh*rx*invrbin;
        const float valslosh = linear_interpolate(rslosh, cpy_sb, nbins);
        img[y*xw+x] += valslosh;
      }
}

void add_sb_prof_multipole(const float rbin, const int nbins, const float *sb,
                           const float xc, const float yc,
                           const int m,
                           const float mag, const double theta0,
                           const int xw, const int yw,
                           float *img)
{
  // copy to ensure outer bin is 0
  std::vector<float> cpy_sb(sb, sb+nbins);
  cpy_sb.push_back(0);

  const float invrbin = 1/rbin;
  const float st0 = float(sin(-theta0*(M_PI/180)));
  const float ct0 = float(cos(-theta0*(M_PI/180)));

  const int x1 = max(int(xc-rbin*nbins), 0);
  const int x2 = min(int(xc+rbin*nbins), xw-1);
  const int y1 = max(int(yc-rbin*nbins), 0);
  const int y2 = min(int(yc+rbin*nbins), yw-1);

  for(int y=y1; y<=y2; ++y)
    for(int x=x1; x<=x2; ++x)
      {
        // calculate rotated dx and dy along theta0 direction
        const float dx = (x-xc)*invrbin;
        const float dy = (y-yc)*invrbin;
        const float rx = dx*ct0 - dy*st0;
        const float ry = dx*st0 + dy*ct0;

        const float r = sqrt(sqr(rx) + sqr(ry));

        float angcmpt = 0;
        if(r > 1.00001f)
          // (hack: 1 here instead of 0 is to fix issue with centre causing large m1)
          // angcmpt = sin(m*theta)
          // below are expansions in terms of sin(theta) and cos(theta),
          // where sin(theta)=ry/r and cos(theta)=rx/r
          switch(m)
            {
            case 1:
              angcmpt = ry/r; break;
            case 2:
              angcmpt = 2*rx*ry / (r*r); break;
            case 3:
              angcmpt = ry*(3/r - 4*ry*ry/(r*r*r)); break;
            case 4:
              angcmpt = 4*rx*ry*(rx*rx - ry*ry)/(r*r*r*r); break;
            }

        const float val = linear_interpolate(r, cpy_sb, nbins) * (1+angcmpt*mag);
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

void resamplePSF(int psf_nx, int psf_ny,
                 double psf_pixsize,
                 double psf_ox, double psf_oy,
                 const float *psf,
                 int img_nx, int img_ny,
                 double img_pixsize,
                 float *img)
{
  const double scale_oi = img_pixsize / psf_pixsize;

  // here pixels go from -0.5->0.5
  double norm = 0;
  for(int y=0; y<img_ny; ++y)
    for(int x=0; x<img_nx; ++x)
      {
        // make symmetric about zero
        const int wrapx = x<img_nx/2 ? x : x-img_nx;
        const int wrapy = y<img_ny/2 ? y : y-img_ny;

        // output rectangle
        const double lxo = wrapx-0.5, uxo = wrapx+0.5, lyo = wrapy-0.5, uyo = wrapy+0.5;

        //printf("Output: (%g,%g) (%g,%g)\n", lxo, lyo, uxo, uyo);

        // calculate input rectangle
        const double lxi = lxo*scale_oi + psf_ox;
        const double uxi = uxo*scale_oi + psf_ox;
        const double lyi = lyo*scale_oi + psf_oy;
        const double uyi = uyo*scale_oi + psf_oy;

        // iterate over pixels and fractional pixels in input
        // ly moves to ny and lx to nx, each time
        double totpix = 0;
        for(double ly=lyi; ly<uyi; )
          {
            const double ny = min(uyi, floor(ly+1));
            for(double lx=lxi; lx<uxi; )
              {
                const double nx = min(uxi, floor(lx+1));
                //printf("  Input: (%g,%g) (%g,%g)\n", lx, ly, nx, ny);

                // addup area contribution if within input psf
                const int ix = int(round(0.5*(nx+lx)));
                const int iy = int(round(0.5*(ny+ly)));
                if(ix>=0 && iy>=0 && ix<psf_nx && iy<psf_ny)
                  {
                    totpix += (ny-ly)*(nx-lx)*double(psf[iy*psf_nx + ix]);
                  }
                lx = nx;
              }
            ly = ny;
          }
        img[y*img_nx + x] = float(totpix);
        norm += totpix;
      }

  // normalise image
  for(int y=0; y<img_ny; ++y)
    for(int x=0; x<img_nx; ++x)
      img[ y*img_nx + x ] /= norm;
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
