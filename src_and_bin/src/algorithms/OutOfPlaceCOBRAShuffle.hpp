#ifndef _OUTOFPLACECOBRASHUFFLE_HPP
#define _OUTOFPLACECOBRASHUFFLE_HPP

#include <algorithm>
#include "BitReversal.hpp"
#include <cstring>

// From Carter and Gatlin 1998
template<typename T, unsigned char LOG_N, unsigned char LOG_BLOCK_WIDTH>
class OutOfPlaceCOBRAShuffle {
public:
  inline static void apply(T*__restrict const v) {
    T*__restrict const result = (T*)malloc(sizeof(T)*1ul<<LOG_N);
    
    constexpr unsigned char NUM_B_BITS = LOG_N - 2*LOG_BLOCK_WIDTH;
    constexpr unsigned long B_SIZE = 1ul << NUM_B_BITS;
    constexpr unsigned long BLOCK_WIDTH = 1ul << LOG_BLOCK_WIDTH;

    T*__restrict buffer = (T*)malloc(sizeof(T)*BLOCK_WIDTH*BLOCK_WIDTH);

    for (unsigned long b=0; b<B_SIZE; ++b) {
      unsigned long b_rev = BitReversal<NUM_B_BITS>::reverse_bytewise(b);
      
      // Copy block to buffer:
      for (unsigned long a=0; a<BLOCK_WIDTH; ++a) {
	unsigned long a_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(a);

	for (unsigned long c=0; c<BLOCK_WIDTH; ++c)
	  buffer[ (a_rev << LOG_BLOCK_WIDTH) | c ] = v[ (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c ];
      }

      // Swap from buffer:
      for (unsigned long c=0; c<BLOCK_WIDTH; ++c) {
      	// Note: Typo in original pseudocode by Carter and Gatlin at
      	// the following line:
      	unsigned long c_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(c);
	
      	for (unsigned long a_rev=0; a_rev<BLOCK_WIDTH; ++a_rev)
      	  result[(c_rev << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b_rev<<LOG_BLOCK_WIDTH) | a_rev] = buffer[ (a_rev<<LOG_BLOCK_WIDTH) | c ];
      }
    }
    free(buffer);

    memcpy(v, result, (1ul<<(LOG_N))*sizeof(T));
    free(result);
  }
};

#endif
