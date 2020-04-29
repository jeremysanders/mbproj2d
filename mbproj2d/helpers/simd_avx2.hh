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

#endif
