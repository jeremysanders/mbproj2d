#ifndef SIMD_SSE2_HH
#define SIMD_SSE2_HH

#include <smmintrin.h>
#include "sse_mathfun.h"

// Floating point wrappers
//////////////////////////

class VecF4
{
public:
  static constexpr size_t nelem = 4;

  VecF4(__m128 x) : v(x)
  {}
  VecF4(float x) : v(_mm_set1_ps(x))
  {}
  VecF4(float a0, float a1, float a2, float a3)
    : v(_mm_set_ps(a0,a1,a2,a3))
  {}
  VecF4(const float* x) : v(_mm_loadu_ps(x))
  {}

  VecF4& operator=(const VecF4& o)
  {
    v = o.v;
    return *this;
  }

  // steps to increment to next block
  static VecF4 step()
  {
    return VecF4(3,2,1,0);
  }

  float operator[](size_t i) const { return v[i]; }
  float& operator[](size_t i) { return v[i]; }

  float hadd() const
  {
    // make this fancier
    return v[0]+v[1]+v[2]+v[3];
  }

  void store(float* out)
  {
    _mm_storeu_ps(out, v);
  }

  __m128 v;
};

inline VecF4 operator+(const VecF4& a, const VecF4& b)
{
  return _mm_add_ps(a.v, b.v);
}

inline VecF4 operator-(const VecF4& a, const VecF4& b)
{
  return _mm_sub_ps(a.v, b.v);
}

inline VecF4 operator*(const VecF4& a, const VecF4& b)
{
  return _mm_mul_ps(a.v, b.v);
}

inline VecF4 operator/(const VecF4& a, const VecF4& b)
{
  return _mm_div_ps(a.v, b.v);
}

inline VecF4 sqrt(const VecF4& a)
{
  return _mm_sqrt_ps(a.v);
}

inline VecF4 log(const VecF4& a)
{
  return log_ps(a.v);
}

inline VecF4 exp(const VecF4& a)
{
  return exp_ps(a.v);
}

inline VecF4 min(const VecF4& a, const VecF4& b)
{
  return _mm_min_ps(a.v, b.v);
}

inline VecF4 max(const VecF4& a, const VecF4& b)
{
  return _mm_max_ps(a.v, b.v);
}

///////////////////////////////
// Integer wrappers

class VecI4
{
public:
  static constexpr size_t nelem = 4;

  VecI4(__m128i x) : v(x)
  {}
  VecI4(int x) : v(_mm_set1_epi32(x))
  {}
  VecI4(int a0, int a1, int a2, int a3)
    : v(_mm_set_epi32(a0,a1,a2,a3))
  {}
  VecI4(const int* x) : v(_mm_loadu_si128
			 (reinterpret_cast<__m128i const *>(x)))
  {}

  VecI4& operator=(const VecI4& o)
  {
    v = o.v;
    return *this;
  }

  // steps to increment to next block
  static VecI4 step()
  {
    return VecI4(3,2,1,0);
  }

  int operator[](size_t i) const { return v[i]; }

  __m128i v;
};

inline VecI4 operator+(const VecI4& a, const VecI4& b)
{
  return _mm_add_epi32(a.v, b.v);
}

inline VecI4 operator-(const VecI4& a, const VecI4& b)
{
  return _mm_sub_epi32(a.v, b.v);
}

inline VecI4 operator*(const VecI4& a, const VecI4& b)
{
  return _mm_mul_epi32(a.v, b.v);
}

// mixed functions

// take a where mask==-1 or b where mask==0
inline VecF4 where(const VecI4& mask, const VecF4& a, const VecF4& b)
{
  return _mm_blendv_ps(a.v, b.v, _mm_castsi128_ps(mask.v));
}

// apply integer mask to floating point value
inline VecF4 and_mask(const VecF4& a, const VecI4& mask)
{
  return _mm_and_ps(a.v, _mm_castsi128_ps(mask.v));
}

#endif
