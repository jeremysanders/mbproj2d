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

#ifndef SIMD_AVX2_HH
#define SIMD_AVX2_HH

#include <immintrin.h>
#include "avx_mathfun.h"

// Floating point wrappers
//////////////////////////

class VecF
{
public:
  static constexpr size_t nelem = 8;

  VecF(__m256 x) : v(x)
  {}
  VecF(float x) : v(_mm256_set1_ps(x))
  {}
  VecF(float a0, float a1, float a2, float a3,
       float a4, float a5, float a6, float a7)
    : v(_mm256_set_ps(a0,a1,a2,a3,a4,a5,a6,a7))
  {}
  VecF(const float* x) : v(_mm256_loadu_ps(x))
  {}

  VecF& operator=(const VecF& o)
  {
    v = o.v;
    return *this;
  }

  // steps to increment to next block
  static VecF step()
  {
    return VecF(7,6,5,4,3,2,1,0);
  }

  float operator[](size_t i) const { return v[i]; }
  float& operator[](size_t i) { return v[i]; }

  float hadd() const
  {
    // make this fancier
    return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
  }

  void store(float* out)
  {
    _mm256_storeu_ps(out, v);
  }

  __m256 v;
};

inline VecF operator+(const VecF& a, const VecF& b)
{
  return _mm256_add_ps(a.v, b.v);
}

inline VecF operator-(const VecF& a, const VecF& b)
{
  return _mm256_sub_ps(a.v, b.v);
}

inline VecF operator*(const VecF& a, const VecF& b)
{
  return _mm256_mul_ps(a.v, b.v);
}

inline VecF operator/(const VecF& a, const VecF& b)
{
  return _mm256_div_ps(a.v, b.v);
}

inline VecF sqrt(const VecF& a)
{
  return _mm256_sqrt_ps(a.v);
}

inline VecF log(const VecF& a)
{
  return log256_ps(a.v);
}

inline VecF exp(const VecF& a)
{
  return exp256_ps(a.v);
}

inline VecF min(const VecF& a, const VecF& b)
{
  return _mm256_min_ps(a.v, b.v);
}

inline VecF max(const VecF& a, const VecF& b)
{
  return _mm256_max_ps(a.v, b.v);
}

///////////////////////////////
// Integer wrappers

class VecI
{
public:
  static constexpr size_t nelem = 8;

  VecI(__m256i x) : v(x)
  {}
  VecI(int x) : v(_mm256_set1_epi32(x))
  {}
  VecI(int a0, int a1, int a2, int a3,
       int a4, int a5, int a6, int a7)
    : v(_mm256_set_epi32(a0,a1,a2,a3,a4,a5,a6,a7))
  {}
  VecI(const int* x) : v(_mm256_loadu_si256
			 (reinterpret_cast<__m256i const *>(x)))
  {}

  VecI& operator=(const VecI& o)
  {
    v = o.v;
    return *this;
  }

  // steps to increment to next block
  static VecI step()
  {
    return VecI(7,6,5,4,3,2,1,0);
  }

  int operator[](size_t i) const { return v[i]; }

  __m256i v;
};

inline VecI operator+(const VecI& a, const VecI& b)
{
  return _mm256_add_epi32(a.v, b.v);
}

inline VecI operator-(const VecI& a, const VecI& b)
{
  return _mm256_sub_epi32(a.v, b.v);
}

inline VecI operator*(const VecI& a, const VecI& b)
{
  return _mm256_mul_epi32(a.v, b.v);
}

// mixed functions

// take a where mask==-1 or b where mask==0
inline VecF where(const VecI& mask, const VecF& a, const VecF& b)
{
  return _mm256_blendv_ps(a.v, b.v, _mm256_castsi256_ps(mask.v));
}

// apply integer mask to floating point value
inline VecF and_mask(const VecF& a, const VecI& mask)
{
  return _mm256_and_ps(a.v, _mm256_castsi256_ps(mask.v));
}

#endif
