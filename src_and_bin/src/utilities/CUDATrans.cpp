#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "MatrixTranspose.h"

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 32



int main()
{
    int i, width, height, nreps, size, wrong, correct;
    double cpuTime, cpuBandwidth;
    cudaError_t cudaStatus;

    double *matrA, *matrATC, *matrATG, *matrAC;

    srand(time(NULL));

    nreps = 10000;
    width = 500;
    height = 100;
    double *dev_matrA = 0;
    double *dev_matrB = 0;
    dim3 dim_grid, dim_block;
    dim_block.x = TILE_DIM;
    dim_block.y = BLOCK_ROWS;
    dim_block.z = 1;

    dim_grid.x = (width + TILE_DIM - 1) / TILE_DIM;
    dim_grid.y = (height + TILE_DIM - 1) / TILE_DIM;
    dim_grid.z = 1;

    size = width * height;

    matrA = (double*)malloc(size * sizeof(double)); // matrix A
    matrAC = (double*)malloc(size * sizeof(double)); // matrix A copied
    matrATC = (double*)malloc(size * sizeof(double)); // matrix A transposed by CPU
    matrATG = (double*)malloc(size * sizeof(double)); // matrix A transposed by GPU

    for (i = 0; i < size; i++)
    {
        matrA[i] = (double)i;
    }


    // copy to gpu
    cudaMemcpy(dev_matrA, matrA, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dev_matrB, 0, size * sizeof(double));
    // call buffered
    buffered_helper << <dim_grid, dim_block >> >(dev_matrB, dev_matrA, width, height, nreps);
    // copy back
    cudaMemcpy(dev_matrA, matrA, size * sizeof(double), cudaMemcpyDeviceToHost);



    cudaMemcpy(dev_matrA, matrA, size * sizeof(double), cudaMemcpyHostToDevice);



}

