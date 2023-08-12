#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ void reverse(char* str, int len)
{
    int i = 0, j = len - 1, temp;
    while (i < j) {
        temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        i++;
        j--;
    }
}

__device__ int intToStr(int x, char str[], int d)
{
    int i = 0;
    while (x) {
        str[i++] = (x % 10) + '0';
        x = x / 10;
    }

    while (i < d)
        str[i++] = '0';

    reverse(str, i);
    str[i] = '\0';
    return i;
}


__device__ void floatToStr(double n, char* res, int afterpoint)
{
    int ipart = (int)n;
    double fpart = n - (double)ipart;
    int i = intToStr(ipart, res, 0);
    if (afterpoint != 0) {
        res[i] = '.';
        fpart = fpart * pow(10, afterpoint);
        intToStr((int)fpart, res + i + 1, afterpoint);
    }
}

__device__ void mySnprintf(char* s, int a, int b, int c, double t)
{
    char temp[50];
    intToStr(a, temp, 0);
    int index = 0;
    int i = 0;
    while(temp[i] != '\0') {
        s[index++] = temp[i++];
    }
    s[index++] = ',';
    s[index++] = ' ';
    i = 0;
    intToStr(b, temp, 0);
    while(temp[i] != '\0') {
        s[index++] = temp[i++];
    }
    s[index++] = ',';
    s[index++] = ' ';
    i = 0;
    intToStr(c, temp, 0);
    while(temp[i] != '\0') {
        s[index++] = temp[i++];
    }
    s[index++] = ' ';
    s[index++] = 's';
    s[index++] = 'a';
    s[index++] = 't';
    s[index++] = 'i';
    s[index++] = 's';
    s[index++] = 'f';
    s[index++] = 'y';
    s[index++] = ' ';
    s[index++] = 'P';
    s[index++] = 'r';
    s[index++] = 'o';
    s[index++] = 'x';
    s[index++] = 'i';
    s[index++] = 'm';
    s[index++] = 'i';
    s[index++] = 't';
    s[index++] = 'y';
    s[index++] = ' ';
    s[index++] = 'C';
    s[index++] = 'r';
    s[index++] = 'i';
    s[index++] = 't';
    s[index++] = 'e';
    s[index++] = 'r';
    s[index++] = 'i';
    s[index++] = 'a';
    s[index++] = ' ';
    s[index++] = 'a';
    s[index++] = 't';
    s[index++] = ' ';
    s[index++] = 't';
    s[index++] = '=';
    s[index++] = ' ';
    floatToStr(t, temp, 2);
    i = 0;
    while(temp[i] != '\0') {
        s[index++] = temp[i++];
    }
    s[index] = '\0';
}


__global__  void checkSatisfiesProximityCriteria2(int currentIndex, double D, int N, int K, double TCount, double* x1, double* x2, double* a, double* b, int* id, char* d_strings)  {
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
       		char* current_string = &d_strings[big_count++];
        	mySnprintf(current_string, satisfiedPoints[0], satisfiedPoints[1], satisfiedPoints[2], t);

	        break; // break out of the loop as soon as we find 3 points
	    }
	}
    }
}


int computeOnGPU(char h_strings[MAX_LEN][MAX_LEN], int currentIndex, double D, int N, int K, double TCount, double* x1, double* x2, double* a, double* b, int* id) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    char* d_strings;
    
    err = cudaMalloc(&d_strings, N * MAX_LEN * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
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
    checkSatisfiesProximityCriteria2<<<blocksPerGrid, threadsPerBlock>>>(currentIndex, D, N, K, TCount, d_x1, d_x2, d_a, d_b, id, d_strings);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();
    err = cudaMemcpy(h_strings, d_strings, MAX_LEN * MAX_LEN * sizeof(char), cudaMemcpyDeviceToHost);
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
    
    if (cudaFree(d_strings) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

