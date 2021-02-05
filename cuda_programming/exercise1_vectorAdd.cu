// 20210122 TIL
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <chrono>
#define NUM_DATA 5029

__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid > NUM_DATA)
		return;
	_c[tid] = _a[tid] + _b[tid];
}

int main(void) {

	std::chrono::system_clock::time_point start, end, total_st, total_end;
	std::chrono::duration<double> timer[5];
	// 0: Total, 1: Kernel, 2: Data host -> device, 3: Data device -> host, 4: CPU vecAdd

	int* a, * b, * c, * h_c;
	int* d_a, * d_b, * d_c;

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes.\n", NUM_DATA, memSize);

	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

	start = std::chrono::system_clock::now();
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}
	end = std::chrono::system_clock::now();
	timer[4] = end - start;

	for (int i = 0; i < NUM_DATA; i++) {
		h_c[i] = a[i] + b[i];
	}

	// 1. CUDA memory allocation
	cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);

	total_st = std::chrono::system_clock::now();
	// 2. Copy input data from CPU memory to GPU memory
	start = std::chrono::system_clock::now();
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
	end = std::chrono::system_clock::now();
	timer[2] = end - start;

	// 3. Kernel execution
	start = std::chrono::system_clock::now();
	int block_size = NUM_DATA / 1024 + 1;
	dim3 dimGrid(block_size);
	dim3 dimBlock(NUM_DATA / block_size + 1);
	vecAdd << < dimGrid, dimBlock >> > (d_a, d_b, d_c);
	cudaDeviceSynchronize();
	end = std::chrono::system_clock::now();
	timer[1] = end - start;

	// 4. Copy output date from GPU memory to CPU memory
	start = std::chrono::system_clock::now();
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
	end = std::chrono::system_clock::now();
	timer[3] = end - start;

	total_end = std::chrono::system_clock::now();
	timer[0] = total_end - total_st;

	// Check result
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (a[i] + b[i] != c[i]) {
			printf("[%d] The result is not matched! (%d, %d)\n",
				i, a[i] + b[i], c[i]);
			result = false;
		}
	}

	for (int i = 0; i < 5; i++)
		timer[i] *= 1000;

	if (result) {
		printf("GPU works well!\n");
		// 0: Total, 1: Kernel, 2: Data host -> device, 3: Data device -> host, 4: CPU vecAdd
		printf("TOTAL[%.10f] KERNEL[%.10f] HOSTtoDEVICE[%.10f] DEVICEtoHOST[%.10f] CPUTOTAL[%.10f]\n",
			timer[0], timer[1], timer[2], timer[3], timer[4]);
	}
	//printf("total_st[%f] total_end[%f]\n", (double)total_st, (double)total_end);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[]a; delete[]b; delete[]c;

	return 0;
}