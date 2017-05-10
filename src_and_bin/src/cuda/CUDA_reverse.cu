
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Unrolled.h"
#include "CUDA_reverse.h"
#include "../utilities/TestClassCOBRA.hpp"
#include "../algorithms/COBRAShuffle.hpp"
#include "assert.h"
#include "TestClassGPU.hpp"
#include "CUDA_naive.h"
#include <complex>

#include <stdio.h>

#define LOG_N 24
#define T std::complex<double>
#define COMPARE_METHOD COBRAShuffle


int main() {
	constexpr unsigned long N = 1ul << LOG_N;

	//These are for time measurements
	TestRuntimeGPU<CudaReversal, T, LOG_N>::run(100, 0.02);
	TestRuntimeGPU<NaiveCudaReversal, T, LOG_N>::run(100, 0.05);
	TestRuntime<COMPARE_METHOD, T, LOG_N, 4>::run(100, 0.05);

	
	//This is the correctness check
	/*T* input_data;
	cudaError_t status = cudaMallocHost((void**)&input_data, N * sizeof(T));
	if (status != cudaSuccess) 
		printf("Error allocating pinned host memory\n");
	T* check_data = new T[N];
	for (unsigned long i = 0; i < N; i++) {
		input_data[i] = T(i);
		check_data[i] = T(i);
	}
	
	CudaReversal<T, LOG_N>::apply(input_data);
	COMPARE_METHOD<T, LOG_N, 2>::apply(check_data);

	for (unsigned long i = 0; i < N; i++) {
		assert(input_data[i] == check_data[i]);
	}*/
	

	printf("%d\n ", LOG_N);
	getchar();
	return;
}
