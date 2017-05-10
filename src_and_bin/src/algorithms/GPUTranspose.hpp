#ifndef _GPUMATRIXTRANSPOSE_HPP
#define _GPUMATRIXTRANSPOSE_HPP

#include <algorithm>

// Note: This code is a good candidate to perform on a GPU.

// Implements cache-oblivious strategy for transposition. In cases
// where recursion can simply be unrolled into a loop (e.g., tall,
// thin matrix or short, wide matrix) it is unrolled for greater
// performance.
  // Base case for recursion; should fit in L1 cache. In practice, it
  // just needs to be large enough to amortize out the cost of the
  // recursions. 
static constexpr unsigned int BLOCK_SIZE = (128 / 8);

  // TODO Cannot call std::swap from HOST on Device
  /*__global__ static void apply_square_naive(double*__restrict const mat, const unsigned long N) {
    for (unsigned long r=0; r<N; ++r)
      for (unsigned long c=r+1; c<N; ++c)
  std::swap(mat[r*N+c], mat[c*N+r]);
  } */

  __global__ static void apply_buffered_naive(double* const dest, const double* __restrict const source, const unsigned long R, const unsigned long C) {
    for (unsigned long r=0; r<R; ++r)
      for (unsigned long c=0; c<C; ++c)
	dest[c*R+r] = source[r*C+c];
  }

  #endif