
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Unrolled.h"
#include "../algorithms/BitReversal.hpp"

#include <stdio.h>

#define THREAD_LIMIT 32

__constant__  int first[] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 25, 27, 29, 31, 35, 37, 39, 43, 47, 55 };
__constant__ int second[] = { 32, 16, 48, 8, 40, 24, 56, 36, 20, 52, 44, 28, 60, 34, 50, 42, 26, 58, 38, 54, 46, 62, 49, 41, 57, 53, 61, 59 };
constexpr int array_length = 28;

template<typename T, unsigned char log_width>
__global__ void kernel_transpose_inplace(T* srcDst)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	int tid_in = (row << log_width) + col;
	int tid_out = (col << log_width) + row;

	//every thread swaps one thing and seems like some of the threads will not even perform swapping
	if (tid_out < tid_in) {
		
		T temp = srcDst[tid_out];
		srcDst[tid_out] = srcDst[tid_in];
		srcDst[tid_in] = temp;
	}
}

template<typename T, int width, int width_half>
__global__ void kernel_transpose_special(T* srcDst)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int matrix_index = row * width;
	int in_col = col % width_half;
	int in_row = col / width_half;
	
	if (in_col > in_row) {

		int tid_in = matrix_index + col;
		int tid_out = matrix_index + in_col * width_half + in_row;
		T temp = srcDst[tid_out];
		srcDst[tid_out] = srcDst[tid_in];
		srcDst[tid_in] = temp;
	}
}

//Only supports 6 bits now
template <typename T, unsigned char NUM_BITS>
__global__ void LocalShuffle(T *input_array, const int blockSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int array_index = id % array_length;
	int tid_in = first[array_index];
	int tid_out = second[array_index];

	T temp = input_array[tid_out];
	input_array[tid_out] = input_array[tid_in];
	input_array[tid_in] = temp;
}

template <typename T, unsigned char NUM_BITS>
void newGpuApply(T * input_array) {
	//here the data is on GPU
	//this part will recurse later
	const unsigned long N = 1ul << NUM_BITS;
	constexpr int width = 1ul << (NUM_BITS / 2);
	constexpr int half_width = 1ul << (NUM_BITS / 4);
	const unsigned long half_size = (N / half_width) *array_length;

	int blockSize;
	int gridSize;

	if (half_size < THREAD_LIMIT) {
		blockSize = half_size;
		gridSize = 1;
	}
	else {
		blockSize = THREAD_LIMIT;
		gridSize = half_size / THREAD_LIMIT;
	}

	dim3 shuffleBlock(blockSize);
	dim3 shuffleGrid(gridSize);

	dim3 block(8, 8);
	dim3 grid;
	grid.x = (width + block.x - 1) / block.x;
	grid.y = (width + block.y - 1) / block.y;

	LocalShuffle<T, NUM_BITS / 2> << < shuffleGrid, shuffleBlock >> >(input_array, half_width);
	kernel_transpose_special<T, width, half_width> << <grid, block >> >(input_array);
	LocalShuffle<T, NUM_BITS / 2> << < shuffleGrid, shuffleBlock >> >(input_array, half_width);

	kernel_transpose_inplace<T, NUM_BITS / 2> << <grid, block >> >(input_array);

	LocalShuffle<T, NUM_BITS / 2> << < shuffleGrid, shuffleBlock >> >(input_array, half_width);
	kernel_transpose_special<T, width, half_width> << <grid, block >> >(input_array);
	LocalShuffle<T, NUM_BITS / 2> << < shuffleGrid, shuffleBlock >> >(input_array, half_width);

}


template <typename T, unsigned char NUM_BITS>
class CudaReversal {
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

		newGpuApply<T, NUM_BITS>(input_on_device);

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
