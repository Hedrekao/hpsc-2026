#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_loadu_ps(x);
  __m512 yvec = _mm512_loadu_ps(y);
  __m512 mvec = _mm512_loadu_ps(m);
  __m512 ones = _mm512_set1_ps(1.0f);
  __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);
    __m512 r2 = _mm512_fmadd_ps(rx, rx, _mm512_mul_ps(ry, ry));
    __mmask16 self = _mm512_cmpeq_epi32_mask(_mm512_set1_epi32(i), lanes);
    r2 = _mm512_mask_blend_ps(self, r2, ones);
    __m512 r2inv = _mm512_rsqrt14_ps(r2);
    __m512 r3inv = _mm512_mul_ps(_mm512_mul_ps(r2inv, r2inv), r2inv);
    __m512 scale = _mm512_mul_ps(mvec, r3inv);
    fx[i] = -_mm512_reduce_add_ps(_mm512_mul_ps(scale, rx));
    fy[i] = -_mm512_reduce_add_ps(_mm512_mul_ps(scale, ry));
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
