#include <stdio.h>
#include <assert.h>
#include <complex>
#include <iostream>
#include <cmath>
#include <iostream>
#include "../utilities/Clock.hpp"


inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(std::complex<double> *odata, const std::complex<double> *idata)
{
  __shared__ std::complex<double> tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

inline static void apply() {
  const int nx = 1024;
  const int ny = 1024;
  const int mem_size = nx*ny*sizeof(std::complex<double>);

  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  int devId = 0;
  checkCuda( cudaSetDevice(devId) );

  std::complex<double> *h_idata = (std::complex<double>*)malloc(mem_size);
  std::complex<double> *h_cdata = (std::complex<double>*)malloc(mem_size);
  std::complex<double> *h_tdata = (std::complex<double>*)malloc(mem_size);
  std::complex<double> *gold    = (std::complex<double>*)malloc(mem_size);
  
  std::complex<double> *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }
    
  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );

  
  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size));
  // warmup
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

  for (int i = 0; i < NUM_REPS; i++)
     transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );


error_exit:
  // cleanup

  checkCuda( cudaFree(d_tdata) );
  checkCuda( cudaFree(d_cdata) );
  checkCuda( cudaFree(d_idata) );
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);

}

static double mean(double * times, int size){
  double sum = 0.0;
  for (int i = 0; i < size; ++i){
    sum += times[i];
  }
  return (sum / (size - 1));
}

static double sd(double * times, int size, double mean){
  double sum = 0.0; 
  for (int i = 0; i < size; ++i){
    sum += std::pow((times[i] - mean),2);
  }
  sum /= (size-1);
  return std::sqrt(sum);
}

int main()
{

  int runs = 100;
  double * times = NULL;
  times = new double[runs];
  Clock c;
  times = new double[runs];
  for (int i = 0; i < runs; ++i ){
      c.tick();
      apply();
      times[i] = c.tock();
  }
  double mean_ = mean(times, runs);

  double sd_ = sd(times, runs, mean_);

  delete [] times;
  std::cout << mean_ << "\t" << sd_ << std::endl;

}