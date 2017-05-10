#ifndef _SEMIRECURSIVESHUFFLE_HPP
#define _SEMIRECURSIVESHUFFLE_HPP

#include "RecursiveShuffle.hpp"

// Identical to RecursiveShuffle but only allows a single recursion
// and broadcasts parallel operations over openmp.
template <typename T, unsigned char NUM_BITS>
class OpenMPSemiRecursiveShuffle {
public:
  inline static void apply(T*__restrict x) {
    // & 1 is the same as % 2:
    if ((NUM_BITS & 1) == 1) {
      // allocate buffer and perform single LSB --> MSB:
      lsb_to_msb<T, NUM_BITS>(x);

      OpenMPSemiRecursiveShuffle<T, NUM_BITS-1>::apply(x);
      OpenMPSemiRecursiveShuffle<T, NUM_BITS-1>::apply(x+(1ul<<(NUM_BITS-1)));
    }
    else {
      constexpr unsigned char SUB_NUM_BITS = NUM_BITS>>1;
      constexpr unsigned long SUB_N = 1ul<<SUB_NUM_BITS;
      
      unsigned long k;

      #pragma omp parallel for
      for (k=0; k<SUB_N; ++k)
      	UnrolledShuffle<T, SUB_NUM_BITS>::apply(x+(k<<SUB_NUM_BITS));

      MatrixTranspose<T>::apply_square(x, SUB_N);

      #pragma omp parallel for
      for (k=0; k<SUB_N; ++k)
      	UnrolledShuffle<T, SUB_NUM_BITS>::apply(x+(k<<SUB_NUM_BITS));
    }
  }
};

// NUM_BITS <= 9, simply use UnrolledShuffle:
template <typename T>
class OpenMPSemiRecursiveShuffle<T, 9> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 9>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 8> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 8>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 7> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 7>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 6> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 6>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 5> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 5>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 4> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 4>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 3> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 3>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 2> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 2>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 1> {
public:
  inline static void apply(T*__restrict x) {
    UnrolledShuffle<T, 1>::apply(x);
  }
};

template <typename T>
class OpenMPSemiRecursiveShuffle<T, 0> {
public:
  inline static void apply(T*__restrict x) {
  }
};

#endif
