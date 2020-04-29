#include <cmath>
#include <cstdio>
#include <algorithm>

#include "simd_avx2.hh"

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

// add a profile to image
// ensure last value in sb is 0

void add_sb_prof(const float rbin, const int nbins, const float *sb,
		 const float xc, const float yc,
		 const int xw, const int yw, float *img)
{
  const int x1 = max(int(xc-rbin*nbins), 0);
  const int x2 = min(int(xc+rbin*nbins), xw-1);
  const int y1 = max(int(yc-rbin*nbins), 0);
  const int y2 = min(int(yc+rbin*nbins), yw-1);

  const float invrbin = 1/rbin;
  
  for(int y=y1; y<=y2; ++y)
    for(int x=x1; x<=x2; ++x)
      {
	const float r = sqrt(sqr(x-xc) + sqr(y-yc)) * invrbin;
	
	const int i0 = min(int(r), nbins-1);
	const int i1 = min(i0+1, nbins-1);
	const float w1 = r-int(r);
	const float w0 = 1-w1;
	
	const float val = sb[i0]*w0 + sb[i1]*w1;

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
      sumdm += data[i]*log(model[i]);
      summ += model[i];
    }
  return sumdm - summ;
}

// AVX Poisson log likelihood (around 2.8x faster than direct version)
float logLikelihoodAVX(int nelem, const float* data, const float* model)
{
  // this is a Kahan summation to improve accuracy
  // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
  VecF sumv(0.f);
  VecF c(0.f); 
  while(nelem >= int(VecF::nelem))
    {
      VecF d(data);
      VecF m(model);
      
      VecF val(d*log(m) - m);
      VecF y = val-c;
      VecF t = sumv+y;
      c = (t-sumv)-y;
      sumv = t;

      nelem -= int(VecF::nelem);
      data += VecF::nelem;
      model += VecF::nelem;
    }

  // add remaining items
  float sum = sumv.hadd();
  for(int i=0; i<nelem; ++i)
    {
      sum += data[i]*log(model[i]) - model[i];
    }
  return sum;
}
