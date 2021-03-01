# 2021/03/01
- CUDA 커널을 통해 CUDA core를 사용할 때에는 각 쓰레드의 인덱스를 변수로 정확히 잡아 접근하는 것이 매우 중요하다.
- 인덱스는 3차원까지 프로그래머가 원하는대로 설정할 수 있고, 각 차원별 최대 인덱스 등은 GPU의 `compute capability`마다 다르므로 이는 NVIDIA documentation을 확인하는 것이 좋다.
- 아래의 예제에서는 각 쓰레드의 인덱스에 접근하는 방법에 대해 알아본다.
- 먼저 예제의 main 함수에서 `dim3`를 통해 블록 차원과 그리드 차원을 선언한 경우, 아래와 같이 x, y, z 변수를 통해 각 차원의 크기를 확인할 수 있다.
- 다음으로 `checkIndex` 커널을 보면, 각 쓰레드의 인덱스에 어떻게 접근하는지 알 수 있다.
- `threadIdx`는 각 블록에 존재하는 쓰레드의 인덱스를 의미한다.
- `threadIdx.x`는 x축 방향의 쓰레드 인덱스를, `threadIdx.y`는 y축 방향의 쓰레드 인덱스를, `threadIdx.z`는 z축 방향의 쓰레드 인덱스를 의미한다.
- `blockIdx`는 `threadIdx`와 비슷하게 각 그리드에 존재하는 블록의 인덱스를 의미한다.
- `blockDim`은 각 블록에 존재하는 쓰레드의 갯수를 의미한다.
- `blockDim.x`는 x축 방향의 쓰레드 갯수를, `blockDim.y`는 y축 방향의 쓰레드 갯수를, `blockDim.z`는 z축 방향의 쓰레드 갯수를 의미한다.
- `gridDim`은 `blockDim`과 비슷하게 각 그리드에 존재하는 블록의 갯수를 의미한다.
- 위와 같이 그리드 내의 블록 갯수, 블록 인덱스, 블록 내의 쓰레드 갯수, 쓰레드 인덱스 등을 사용하여 커널에서 생성한 모든 쓰레드에 접근할 수 있다.
- 예를 들어, 커널에서 x축 방향의 다수의 블록 내에 x축 방향으로 1차원으로 쓰레드를 선언한 경우 각 쓰레드에 접근할 때에는 아래와 같이 선언하면 된다.
``` c
__global__ void Index1Dim(void) {
    int thx = blockDim.x * blockIdx.x + threadIdx.x;
}
```
- 또 커널에서 1개의 블록 내에 2차원으로 쓰레드를 선언하고 (m, n)에 해당하는 쓰레드에 접근할 때에는 아래와 같이 선언하면 된다.
``` c
__global__ void Index2Dim(void) {
	int idx = blockDim.x * threadIdx.y + threadIdx.x;
}
```
- 위와 비슷한 방식으로 다수의 블록과 다수의 쓰레드를 선언할 경우 `gridDim` 및 `blockDim`, `blockIdx`, `threadIdx`를 조합하여 각 쓰레드에 접근하면 된다.
- 아래는 이번 예제의 전체 코드이다.
``` c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void checkIndex(void) {
	printf("threadIdx(%d, %d, %d) blockIdx(%d, %d, %d) blockDim(%d, %d, %d) gridDim(%d, %d, %d)\n",
		threadIdx.x, threadIdx.y, threadIdx.z,
		blockIdx.x, blockIdx.y, blockIdx.z,
		blockDim.x, blockDim.y, blockDim.z,
		gridDim.x, gridDim.y, gridDim.z);
}

int main(void) {
	int nElem = 6;
	dim3 block(3, 2, 2);
	dim3 grid((nElem + block.x - 1) / block.x, 2, 1);

	printf("grid.x [%d] grid.y [%d] grid.z [%d]\n", grid.x, grid.y, grid.z);
	printf("block.x[%d] block.y[%d] block.z[%d]\n", block.x, block.y, block.z);

	checkIndex <<<grid, block >>> ();
	cudaDeviceReset();

	return 0;
}
```