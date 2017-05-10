
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Unrolled.h"
#include "../algorithms/BitReversal.hpp"

#include <stdio.h>

#define THREAD_LIMIT 64

template<typename T, unsigned char LOG_N>
__device__ inline static unsigned long reverse_bitwise(unsigned long x) {
	unsigned long maskFromLeft = 1 << LOG_N;
	unsigned long res = 0;
	unsigned int bitNum = LOG_N;
	while (maskFromLeft > 0) {
		unsigned char bit = (x & maskFromLeft) >> bitNum;
		res |= (bit << (LOG_N - 1 - bitNum));
		--bitNum;
		maskFromLeft >>= 1;
	}
	return res;
}


template<typename T, unsigned char LOG_N>
__global__ void kernel_swap_inplace(T* srcDst)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int reversed = reverse_bitwise<T, LOG_N>(index);

	//every thread swaps one thing and seems like some of the threads will not even perform swapping
	if (reversed < index) {

		T temp = srcDst[index];

		srcDst[index] = srcDst[reversed];
		srcDst[reversed] = temp;
	}
}

template <typename T, unsigned char NUM_BITS>
void naiveGpuApply(T * input_array) {
	//here the data is on GPU
	//this part will recurse later
	const unsigned long width = 1ul << (NUM_BITS / 2);

	int blockSize;
	int gridSize;

	if (width < THREAD_LIMIT) {
		blockSize = width;
		gridSize = 1;
	}
	else {
		blockSize = THREAD_LIMIT;
		gridSize = width / THREAD_LIMIT;
	}

	kernel_swap_inplace<T, NUM_BITS> << <gridSize, blockSize >> >(input_array);

}

template <typename T, unsigned char NUM_BITS>
class NaiveCudaReversal {
public:
	//  __attribute__((always_inline))
	static void apply(T *__restrict input_on_host) {
		//move the data to gpu
		const unsigned long N = 1ul << NUM_BITS;

		T *input_on_device = 0;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&input_on_device, N * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(input_on_device, input_on_host, N * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		naiveGpuApply<T, NUM_BITS>(input_on_device);

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(input_on_host, input_on_device, N * sizeof(T), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		//is this evil
	Error:
		cudaFree(input_on_device);
		return;
	}


};
