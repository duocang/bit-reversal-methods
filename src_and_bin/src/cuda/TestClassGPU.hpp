#include "../utilities/Clock.hpp"
#include <iostream>

#include <utility>
#include "cuda_runtime.h"

#include<math.h>
#include<utility>

//I had to move this here even though it is literally the same as utils in order to avoid compile errors
namespace utilsGpu {
	double mean(double * values, int size) {
		double sum{ 0.0 };
		for (int i = 0; i < size; ++i)
			sum += values[i];
		return (sum / size);
	}

	std::pair<int, int> get_indices(int size, float quantile) {
		float pos = size * quantile;
		std::pair<int, int> indices(floor(pos), ceil(pos));
		return indices;
	}

	std::pair<double, double> errors(double * values, int size, float quantile) {
		// sort values in-place
		std::sort(values, values + size);
		// determine indices of lower quantile
		std::pair<int, int> indices = get_indices(size, quantile);
		std::pair<double, double> errors;

		errors.first = (values[indices.first] + values[indices.second]) / 2.0;

		// now determine indices of upper quantile
		indices = get_indices(size, 1.0 - quantile);

		errors.second = (values[indices.first] + values[indices.second]) / 2.0;

		return errors;
	}

}

template<template<typename , unsigned char> class Shuffler, typename T, unsigned char LOG_N>
class TestRuntimeGPU{
public:
	static void run(int runs, float quant){
		Clock c;
		const unsigned long N = 1ul<<LOG_N;
		T* testArray;
		cudaError_t status = cudaMallocHost((void**)&testArray, N * sizeof(T));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory\n");
		double * times = new double[runs];
		for (int i = 0; i < runs; ++i ){
			c.tick();
			Shuffler<T, LOG_N>::apply(testArray);
			times[i] = c.tock();
		};

		double mean_ = utilsGpu::mean(times, runs);

		std::pair<double, double> errors = utilsGpu::errors(times, runs, quant);

		delete [] times;
		std::cout << mean_ << '\t' << errors.first << '\t' << errors.second << std::endl;
	}
};
