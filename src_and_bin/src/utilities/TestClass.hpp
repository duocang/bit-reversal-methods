#include "Clock.hpp"
#include <iostream>
#include "TestUtils.hpp"
#include <utility>

template<template<typename , unsigned char> class Shuffler, typename T, unsigned char LOG_N>
class TestRuntime{
public:
	static void run(int runs, float quant){
		Clock c;
		const unsigned long N = 1ul<<LOG_N;
		T*testArray = new T[N];
		double * times = new double[runs];
		for (int i = 0; i < runs; ++i ){
			c.tick();
			Shuffler<T, LOG_N>::apply(testArray);
			times[i] = c.tock();
		};

		double mean_ = utils::mean(times, runs);

		std::pair<double, double> errors = utils::errors(times, runs, quant);

		delete [] times;
		std::cout << mean_ << '\t' << errors.first << '\t' << errors.second << std::endl;
	}
};
