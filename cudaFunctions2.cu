#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include "myProto.h"

__global__  void changeT(double* t, int currentIndex, double TCount, int sizeT)  {
    int i = currentIndex + blockDim.x * blockIdx.x + threadIdx.x;
    if(i < sizeT) {
        t[i] = sin(t[i] * M_PI / 2.0);
    }
}

__global__  void checkSatisfiesProximityCriteria3(double* x, double* y, double* t, int currentIndex, int N, double TCount, double* x1, double* x2, double* a, double* b)  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = currentIndex + blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) {
        x[i] = ((x2[i] - x1[i]) / 2.0) * t[j] + (x2[i] + x1[i]) / 2.0;
        y[i] = a[i] * x[i] + b[i];
    }

}


int computeOnGPU(double* x, double* y, double* t, int sizeT, int currentIndex, int N, double TCount, double* x1, double* x2, double* a, double* b) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    double* d_x;
    double* d_y;
    size_t nSize = N * sizeof(double);
    size_t tSize = sizeT * sizeof(double);

    // Allocate memory for each pointer in the array
    err = cudaMalloc((void**)&d_x, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_results - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_y, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  

    // Allocate memory on GPU to copy the data from the host
    double *d_x1, *d_x2, *d_a, *d_b, *d_t;
    err = cudaMalloc((void **)&d_x1, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_x2, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_a, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_b, nSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_t, tSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_x1, x1, nSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_x2, x2, nSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_a, a, nSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_b, b, nSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 

    err = cudaMemcpy(d_t, t, tSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (TCount + threadsPerBlock - 1) / threadsPerBlock;
    changeT<<<blocksPerGrid, threadsPerBlock>>>(d_t, currentIndex, TCount, sizeT);
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    checkSatisfiesProximityCriteria3<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_t, currentIndex, N, TCount, d_x1, d_x2, d_a, d_b);

    cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();

    err = cudaMemcpy(x, d_x, nSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device to host for d_x - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    err = cudaMemcpy(y, d_y, nSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host for d_y -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free allocated memory on GPU
    if (cudaFree(d_x1) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_x2) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_a) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_b) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_x) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    if (cudaFree(d_y) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_t) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}


