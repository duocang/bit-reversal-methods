
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Unrolled.h"

#include <stdio.h>

cudaError_t reverseBlocksWithCUDA(int *input_host, unsigned int array_size, unsigned int block_size);

__global__ void sortBlock(int *input_array, const int blockSize)
{
	int id = threadIdx.x;
	int start_index = id * blockSize;

	UnrolledShuffle<int, 4>::apply(&input_array[start_index]);

}

int main()
{
    const unsigned int array_size = 256;
	const unsigned int block_size = 16;
    int input_array[array_size] = { 0 };

	for (int i = 0; i < array_size; i++){
		input_array[i] = i;
	}


    cudaError_t cudaStatus = reverseBlocksWithCUDA(input_array, array_size, block_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t reverseBlocksWithCUDA(int *input_host, unsigned int array_size, unsigned int block_size)
{
    int *input_device = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&input_device, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(input_device, input_host, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	int core_count = array_size / block_size;
	sortBlock <<< 1, core_count >>>(input_device, block_size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(input_host, input_device, array_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	for (int i = 0; i < core_count; i++) {
		for (int j = 0; j < block_size; j++) {
			printf("%d ", input_host[i * block_size + j]);
		}
		printf("\n");
	}

Error:
    cudaFree(input_device);
    
    return cudaStatus;
}