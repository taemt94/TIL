// GPU에는 크게 글로벌 메모리, 공유(shared) 메모리, 레지스터로 3가지의 메모리가 존재한다.
// 글로벌 메모리는 말 그대로 전역 메모리로 모든 블록의 모든 쓰레드에서 접근 가능하고 용량도 가장 크지만, 
// 그만큼 접근 시간은 가장 길다.
// 공유 메모리는 각각의 블록에서만 접근 가능한 메모리로, 전역 메모리보다 용량은 작지만, 접근 시간은 짧다는 장점이 있다.
// 공유 메모리의 용량은 GPU 모델마다 차이가 있으므로, 사용하는 GPU의 공유 메모리 용량을 확인한 후 사용하는 것이 좋다.
// 코딩을 해본 결과 나의 코드에서 공유 메모리에 변수를 선언할 때 공유 메모리 용량을 초과할 경우 컴파일러가 컴파일할 때
// 에러를 발생시키는 것을 확인하였다.
// 레지스터는 각각의 쓰레드마다 할당된 메모리로 3가지 메모리 중 용량은 가장 작은 반면, 접근 시간은 가장 짧다.
// GPU의 경우 위와 같이 3가지 메모리가 존재하고, 각 메모리마다 용량 및 접근 시간이 다르므로 코딩을 할 때 이를 고려하여
// 코딩을 해야 효율적인 코드를 작성할 수 있다.
// 예를 들어, 모든 변수를 글로벌 메모리에 선언할 경우 접근 시간이 길어지므로 연산 시간이 늘어날 수 있고, 공유 메모리나 레지스터를
// 잘 활용할 경우 훨씬 효율적으로 코드를 작성할 수도 있다.

// 아래의 예제는 글로벌 메모리만을 사용할 때와 공유 메모리를 사용할 때를 비교하는 예제이다.
// 공유 메모리를 선언할 때에는 커널 내에서 __shared__ 를 붙여서 변수를 선언하면 된다.
// 공유 메모리를 사용할 때 고려해야 할 점은 CPU 메모리에서 GPU의 글로벌 메모리로 복사한 메모리를 커널 상에서 다시 공유 메모리로
// 복사해주어야 한다는 점이다.
// 따라서 이러한 추가적인 복사 과정을 감수할 만큼 많이 접근하는 변수인지를 판단해보아야 한다.
// 추가적으로 GPU 모델마다 공유 메모리를 접근하는 오버헤드가 다른 것을 확인하였다.
// 강의를 들으면서 강사님의 GPU 상에서 돌릴 때에는 공유 메모리를 사용하는 것이 훨씬 연산 시간이 빨랐지만,
// 내 컴퓨터에서 돌릴 때에는 오히려 글로벌 메모리만을 사용하였을 때 연산 시간이 빨랐다.
// 따라서 이러한 경우가 존재할 수 있음을 항상 고려사항으로 염두해두고 프로그래밍을 해야 할 것 같다.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

#define ROW_SIZE 32
#define COL_SIZE 32
#define K_SIZE 128
#define WORK_LOAD 1024

__global__ void matMul_kernel(float* _A, float* _B, float* _C) {
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++) {
		for (int w = 0; w < WORK_LOAD; w++)
			_C[index] += _A[row * K_SIZE + k] * _B[k * COL_SIZE + col];
	}
}

// Using shared memory sA, sB
__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C) {
	__shared__ float sA[ROW_SIZE][K_SIZE];
	__shared__ float sB[K_SIZE][COL_SIZE];

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int k = 0; k < K_SIZE; k++) {
		sA[row][k] = _A[row * K_SIZE + k];
		sB[k][col] = _B[k * COL_SIZE + col];
	}
	__syncthreads();

	int index = row * blockDim.x + col;
	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++) {
		for (int w = 0; w < WORK_LOAD; w++)
			_C[index] += sA[row][k] * sB[k][col];
	}
}

int main(void) {
	std::chrono::system_clock::time_point st, end, st1, end1;
	std::chrono::duration<double> cpu_time, gpu_time, sgpu_time;
	std::chrono::duration<double> gpu_k_time, sgpu_k_time;

	float A[ROW_SIZE][K_SIZE] = { 0, };
	float B[K_SIZE][COL_SIZE] = { 0, };
	float C[ROW_SIZE][COL_SIZE] = { 0, };
	float cudaC[ROW_SIZE][COL_SIZE] = { 0, };
	float scudaC[ROW_SIZE][COL_SIZE] = { 0, };

	int AmemSize = sizeof(float) * (ROW_SIZE * K_SIZE);
	int BmemSize = sizeof(float) * (COL_SIZE * K_SIZE);
	int CmemSize = sizeof(float) * (ROW_SIZE * COL_SIZE);

	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < K_SIZE; j++) {
			A[i][j] = i;
			B[j][i] = j;
		}

	// CPU matMul
	st = std::chrono::system_clock::now();
	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < COL_SIZE; j++)
			for (int k = 0; k < K_SIZE; k++)
				for (int w = 0; w < WORK_LOAD; w++)
					C[i][j] += A[i][k] * B[k][j];
	end = std::chrono::system_clock::now();
	cpu_time = end - st;

	// 1. Cuda memory allocation
	float* cA, * cB, * cC;
	cudaMalloc(&cA, AmemSize);
	cudaMalloc(&cB, BmemSize);
	cudaMalloc(&cC, CmemSize);

	// 2. Copy input data from host to device
	st = std::chrono::system_clock::now();
	cudaMemcpy(cA, A, AmemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cB, B, BmemSize, cudaMemcpyHostToDevice);

	// 3. kernel execution
	st1 = std::chrono::system_clock::now();
	dim3 gDim(1);
	dim3 bDim(COL_SIZE, ROW_SIZE);
	matMul_kernel << <gDim, bDim >> > (cA, cB, cC);
	cudaDeviceSynchronize();
	end1 = std::chrono::system_clock::now();
	gpu_k_time = end1 - st1;

	// 4. Copy output data from device to host
	cudaMemcpy(cudaC, cC, CmemSize, cudaMemcpyDeviceToHost);

	end = std::chrono::system_clock::now();
	gpu_time = end - st;

	// Check result
	bool flag = true;
	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < COL_SIZE; j++) {
			if (C[i][j] != cudaC[i][j]) {
				printf("C[%d][%d]: %f, cudaC[%d][%d]: %f\n",
					i, j, C[i][j], i, j, cudaC[i][j]);
				flag = false;
			}
		}

	if (flag)
		printf("Well done, GPU!\n");

	printf("CPU[%f]\n", cpu_time);
	printf("GPU[%f]\n", gpu_time);

	cudaFree(cA); cudaFree(cB); cudaFree(cC);


	// Using shared memory sA, sB
	// 1.
	float* scA, * scB, * scC;
	cudaMalloc(&scA, AmemSize);
	cudaMalloc(&scB, BmemSize);
	cudaMalloc(&scC, CmemSize);

	// 2.
	st = std::chrono::system_clock::now();

	cudaMemcpy(scA, A, AmemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(scB, B, BmemSize, cudaMemcpyHostToDevice);

	// 3.
	st1 = std::chrono::system_clock::now();
	dim3 sgDim(1);
	dim3 sbDim(COL_SIZE, ROW_SIZE);
	matMul_kernel_shared << <sgDim, sbDim >> > (scA, scB, scC);
	cudaThreadSynchronize();

	end1 = std::chrono::system_clock::now();
	sgpu_k_time = end1 - st1;

	// 4.
	cudaMemcpy(scudaC, scC, CmemSize, cudaMemcpyDeviceToHost);

	end = std::chrono::system_clock::now();
	sgpu_time = end - st;

	// Check result
	bool sflag = true;
	for (int i = 0; i < ROW_SIZE; i++)
		for (int j = 0; j < COL_SIZE; j++) {
			if (C[i][j] != scudaC[i][j]) {
				printf("C[%d][%d]: %f, cudaC[%d][%d]: %f\n",
					i, j, C[i][j], i, j, scudaC[i][j]);
				sflag = false;
			}
		}

	if (sflag)
		printf("Well done, shared GPU!\n");

	printf("shared GPU[%f]\n", sgpu_time);
	printf("gpu_kernel[%f] sgpu_kernel[%f]\n",
		gpu_k_time, sgpu_k_time);

	cudaFree(scA); cudaFree(scB); cudaFree(scC);
}