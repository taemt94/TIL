# 02/05
- 산학 협력 과제에서 CUDA 프로그래밍을 통한 GPU 병렬 알고리즘을 개발하게 되어 CUDA 프로그래밍을 온라인 강의를 수강하며 공부하였고, 배운 내용을 간단하게 정리하려 한다.
- 참고한 온라인 강의는 아래의 링크와 같다.  
   <https://hpclab.tistory.com/4>

- 아래의 예제는 CUDA 커널을 이용하여 벡터의 합을 계산하는 코드이다.
- CUDA 프로그래밍은 아래와 같이 크게 4가지 작업으로 구성된다.
	1. GPU 상에 메모리 할당하기
	2. CPU 상의 변수를 GPU 상의 변수로 복사하기
	3. CUDA 커널(함수)을 이용하여 GPU 상에서 연산하기
	4. 연산된 결과를 다시 GPU에서 CPU로 복사하기
- 각 작업에 대해 자세히 살펴보면 다음과 같다.
    1. GPU 상에 메모리 할당하기
        - `cudaMalloc()` 함수를 사용하여 GPU 상에 메모리를 할당한다.
        - 사용 방법은 C에서 `malloc()` 함수를 사용하는 방법과 비슷하여 쉽게 사용할 수 있다. 
       	``` c++
       	// 1. CUDA memory allocation
       	int memSize = sizeof(int) * NUM_DATA; // NUM_DATA: 1024
       	cudaMalloc(&d_a, memSize);
       	cudaMalloc(&d_b, memSize);
       	cudaMalloc(&d_c, memSize);
       	```

    2. CPU 상의 변수를 GPU 상의 변수로 복사하기
        - `cudaMemcpy()` 함수를 사용하여 CPU 상의 변수 데이터를 GPU 상의 변수에 복사해준다.
        - 마찬가지로 사용 방법은 C에서 `memcpy()` 함수를 사용하는 방법과 비슷하다.
     	- 한가지 다른 점은 마지막 인자로 메모리가 어디에서 어디로 이동할 것인지 방향성을 알려줄 수 있는 입력을 주어야 한다는 것이다.
     	- 이는 아마 CUDA 컴파일러가 컴파일을 할 때 데이터가 어느 메모리에 있는지 알 수 없기 때문인 것으로 보인다.
     	- 종류로는 `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyHostToHost`, `cudaMemcpyDeviceToDevice` 등이 있고, 데이터를 복사할 때 필요한 인자를 입력하면 된다.
  		``` c++
  		// 2. Copy input data from CPU memory to GPU memory
       	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
       	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
  		```

    3. CUDA 커널(함수)을 이용하여 GPU 상에서 연산하기
        - CUDA 커널을 호출할 때에는 아래와 같은 문법을 사용한다.
        - 먼저 함수 이름을 적고, `<<< >>>` 안에 사용할 블록 차원과 쓰레드 차원을 적은 후 마지막으로 커널에 입력할 인자를 괄호안에 적어주면 된다.
        - CUDA 커널을 호출하게 되면, GPU 상의 쓰레드를 사용할 수 있는 상태가 된다.
        - 먼저 `dimGrid` 변수를 통해 블록을 (block_size, 1, 1)의 1차원으로 2개의 블록을 사용하도록 지정하였다.
        - 또 `dimBlock` 변수를 통해 1개의 블록 당 `(NUM_DATA / block_size + 1)`개의 쓰레드가 1차원으로 생성되도록 하였다.
        - 간단한 예제를 해보기 위해 그리드와 블록의 차원을 1차원으로 하였지만, 필요에 따라 차원을 3차원까지 설정할 수 있다.
        - 가장 아래줄의 `cudaDeviceSynchronize()` 함수는 GPU 상의 커널이 연산을 모두 완료할 때까지 CPU가 대기하도록 해주는 함수이다.
        - GPU와 CPU는 서로 다른 소프트웨어이므로 서로가 어떤 연산을 어디까지 수행했는지 알 수 없고, CPU는 CUDA 커널을 호출하여 GPU 상에서 커널 연산이 수행되도록 한 후 곧바로 다음 작업으로 넘어간다.
        - 이러한 경우 GPU 상에서 커널 연산이 완료되지도 않았는데 CPU 상에서 GPU 상의 메모리를 복사하려고 시도하는 등의 문제가 발생하게 된다.
        - 따라서 커널 뒤에 `cudaDeviceSynchronize()` 함수를 사용함으로써 커널 연산이 모두 완료될 때까지 해당 위치에서 CPU가 대기하도록 하여 위와 같은 문제를 해결한다고 한다.
        - 참고로, `cudaMalloc()`이나 `cudaMemcpy()`의 경우에는 자동으로 CPU가 작업이 끝날 때까지 대기 상태에 있기 때문에 `cudaDeviceSynchronize()`를 사용할 필요가 없다고 한다.
  		``` c++
  		// 3. Kernel execution
       	int block_size = NUM_DATA / 1024 + 1;
       	dim3 dimGrid(block_size);
       	dim3 dimBlock(NUM_DATA / block_size + 1);
       	vecAdd <<< dimGrid, dimBlock >>> (d_a, d_b, d_c);
       	cudaDeviceSynchronize();
  		```

    4. 연산된 결과를 다시 GPU에서 CPU로 복사하기
        - 마지막 작업은 GPU 상에서 CUDA 커널을 통해 연산한 결과를 다시 CPU 메모리로 복사해주는 것이다.
        - CPU에서 GPU로 메모리를 복사할 때와는 반대로 `cudaMemcpyDeviceToHost` 인자를 입력해주면 된다.
  		``` c++
          // 4. Copy output date from GPU memory to CPU memory
       	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
  		```

- CUDA 커널
    - CUDA 커널을 선언할 때에는 기존의 함수 앞에 `__global__`을 붙여주면 된다.
    - `__global__`은 GPU 상에서 사용하는 함수이지만, CPU 상에서 호출하겠다는 의미이다.
    - `__global__`외에도 `__host__`, `__divice__`, `__host__ __device__` 등 다양한 인자가 존재하는데 이는 추후에 정리해보는 것으로 한다.
	- GPU의 커널 연산은 여러 개의 쓰레드를 사용하기 때문에 원하는 쓰레드의 인덱스를 제대로 찾아 접근하는 것이 매우 중요하다.
    - 아래의 코드에서 `blockDim.x`는 1개의 블록 내에 있는 쓰레드의 x축 방향 갯수를 의미한다.
    - 또 `blockIdx.x`는 블록의 x축 방향으로의 인덱스(0, 1, ...)를 의미한다.
    - `blockIdx.x * blockDim.x`를 하면, 각 블록의 0번째 쓰레드의 인덱스에 접근하게 된다.
    - 그리고 `threadIdx.x`는 1개의 블록 내에 있는 쓰레드의 x축 방향으로의 인덱스(0, 1, ...)를 의미한다.
    - 따라서 `blockIdx.x * blockDim.x`에 `threadIdx.x`를 더해주면, CUDA 커널로 생성된 모든 쓰레드를 인덱스로 접근이 가능하다.
    - 아래의 코드에 있는 조건문의 경우 예외 처리를 해주는 코드이다.
    - 1개의 블록 내에는 최대 1024개의 쓰레드를 생성할 수 있고, 쓰레드를 생성할 때에 여러 블록이 서로 다른 갯수의 쓰레드를 생성하도록 하는 것은 불가능하다.
    - 예를 들어 배열의 크기가 1025이면 1025개의 쓰레드를 1개의 블록 내에 생성하는 것은 불가하므로, 2개의 블록을 사용하여야 한다.
    - 그러나 1개의 블록은 1024개의 쓰레드를, 나머지 1개의 블록은 1개의 쓰레드를 가지도록 하는 것 또한 불가하므로, 총 2개의 블록에 2048개의 쓰레드가 생성된다.
    - 이렇게 되면, 2048 - 1025 = 1023개의 쓰레드는 배열의 크기와 맞지 않으므로 1023개의 쓰레드의 인덱스를 통해 배열에 접근하려고 시도하면 `Segmemtation fault`가 발생할 것이다.
    - 이러한 경우를 방지하기 위해 쓰레드의 인덱스가 배열의 크기를 초과하게 될 경우 아무 작업을 하지 않고 `return`하도록 하기 위해 조건문을 추가한 것이다.
    - 쓰레드를 생성할 때 위와 같은 제약 조건이 존재하기 때문에 조건문을 통한 예외 처리도 CUDA 프로그래밍에 있어서 매우 중요하다.
      ``` c++
      __global__ void vecAdd(int* _a, int* _b, int* _c) {
      	int tid = blockIdx.x * blockDim.x + threadIdx.x;
      	if (tid > NUM_DATA)
      		return;
      	_c[tid] = _a[tid] + _b[tid];
      }
      ```
     
- 아래는 전체 소스 코드이다.
- 추가적으로 코드를 보면 연산 결과를 확인하는 코드가 있다.
- 커널 내에서 쓰레드의 인덱스를 사용하여 연산을 수행하다 보면, 연산 결과가 CPU 상에서의 연산 결과와 다르게 나오는 경우가 발생할 수 있으므로, 두 연산 결과가 일치하는지를 확인하는 코드를 항상 추가해주는 것이 좋다.
- 또한, `cudaMalloc()`을 통해 GPU 상에서도 메모리를 동적 할당한 것이므로 함수가 종료되기 전 `cudaFree()`를 통해 할당된 메모리를 다시 반환해주어야 한다.
``` c++
// 20210122 TIL
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <chrono>
#define NUM_DATA 1024

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

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[]a; delete[]b; delete[]c;

	return 0;
}
```



