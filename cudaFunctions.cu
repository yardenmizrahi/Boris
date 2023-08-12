#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__global__  void checkSatisfiesProximityCriteria2(int currentIndex, double D, int N, int K, double TCount, double* x1, double* x2, double* a, double* b, int* id, int* d_results[3], double* d_t_results)  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int big_count = 0;
    double t = 2.0 * i / TCount - 1.0;

    int satisfiedPointsCount = 0;
    int satisfiedPoints[3]; // To store the IDs of points that satisfy the proximity criteria.

    for (int j = 0; j < N; j++) {
    	int res;
	double x_1, x_2, y_1, y_2;
    	int count = 0;

    	for (int h = 0; h < N; h++) {
		if (h != currentIndex) {
			x_1 = ((x2[currentIndex] - x1[currentIndex]) / 2.0) * sin(t * M_PI / 2.0) + (x2[currentIndex] + x1[currentIndex]) / 2.0;
			x_2 = ((x2[h] - x1[h]) / 2.0) * sin(t * M_PI / 2.0) + (x2[h] + x1[h]) / 2.0;
			y_1 = a[currentIndex] * x_1 + b[currentIndex];
			y_2 = a[h] * x_2 + b[h];
		    double distance = sqrt((x_1 - x_2)*(x_1 - x_2) + (y_1 - y_2)*(y_1 - y_2));
		    if (distance < D) {
		        count++;
		    }
		}
	    }

    	res = count >= K; // returns 1 (true) if the criteria is satisfied, otherwise 0 (false)

	if (res) {
	    satisfiedPoints[satisfiedPointsCount] = id[j];
	    satisfiedPointsCount++;

	    if (satisfiedPointsCount == 3) {
       		int current_res[3] = {satisfiedPoints[0], satisfiedPoints[1], satisfiedPoints[2]};
            d_results[big_count] = current_res;
            d_t_results[big_count++] = t;
	        break; // break out of the loop as soon as we find 3 points
	    }
	}
    }
}


int computeOnGPU(int* results[3], double* t_results, int currentIndex, double D, int N, int K, double TCount, double* x1, double* x2, double* a, double* b, int* id) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    int* d_results[3];
    double* d_t_results;
    
    err = cudaMalloc((void**)d_results, N * 3 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_results, 0, N * 3 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device memory to zero - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_t_results, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_t_results, 0, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device memory to zero - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
}
	
  

    // Allocate memory on GPU to copy the data from the host
    double *d_x1, *d_x2, *d_a, *d_b;
    int* d_id;
    err = cudaMalloc((void **)&d_x1, N * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_x2, N * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_a, N * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_b, N * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_id, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_x1, x1, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_x2, x2, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_id, id, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    checkSatisfiesProximityCriteria2<<<blocksPerGrid, threadsPerBlock>>>(currentIndex, D, N, K, TCount, d_x1, d_x2, d_a, d_b, id, d_results, d_t_results);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();
    err = cudaMemcpy(results, d_results, N * 4 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(t_results, d_t_results, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
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
    
    if (cudaFree(d_t_results) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

