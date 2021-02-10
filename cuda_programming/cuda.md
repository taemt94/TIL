# 0128
### CUDA PROGRAMMING
``` c
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int a_value = 3;
__device__ int b_value = 5;

__global__ void test_exchange(int *a, int *b){
    printf("===Kernel start===\n");
    printf("BEFORE a_value[%d] b_value[%d]\n", a_value, b_value);
    printf("BEFORE a_gpu[%d] b_gpu[%d]\n", *a, *b);
    atomicExch(&a_value, a);
    *b = atomicExch(&b_value, b);
    printf("AFTER a_value[%d] b_value[%d]\n", a_value, b_value);
    printf("AFTER a_gpu[%d] b_gpu[%d]\n", *a, *b);
    printf("===Kernel end  ===\n");
}

int main(void){
    int *a_cpu, *b_cpu;
    a_cpu = new int(1);
    b_cpu = new int(2);
    printf("BEFORE a_cpu[%d] b_cpu[%d]\n", a_cpu, b_cpu);

    int memSize = sizeof(int);
    int *a_gpu, *b_gpu;
    cudaMalloc(&a_gpu, memSize); cudaMalloc(&b_gpu, memSize);
    cudaMemcpy(a_gpu, a_cpu, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, memSize, cudaMemcpyHostToDevice);

    test_exchange<<<1, 1>>>(a_gpu, b_gpu);

    cudaMemcpy(a_cpu, a_gpu, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_gpu, memSize, cudaMemcpyDeviceToHost);

    printf("AFTER a_cpu[%d] b_cpu[%d]\n", a_cpu, b_cpu);

    cudaFree(a_gpu); cudaFree(b_gpu);
    delete a_cpu; delete b_cpu; 
}
```
![RESULT](../2021_01/img1.png)

- atomicExch 함수를 테스트해보았다.
- atomicExch 함수는 race condition이 생기지 않도록 하면서 입력한 두개의 값을 교환한다.
- 출력 결과를 보면, a_value 값은 3에서 1로 바뀌었지만, a_gpu 값은 그대로 1인 것을 알 수 있다.
- 이는 atomicExch 함수가 첫번째 인자의 값만 두번째 인자의 값으로 바꾸기 때문이다.
- 또한 atomicExch 함수는 첫번째 인자의 바꾸기 전의 값을 리턴한다.
- 따라서 두 변수의 값을 말그대로 교환하고 싶으면 `*b = atomicExch(&b_value, b);` 와 같이 바꿀 값을 첫번째 인자에 넣어 그 값을 리턴받는 형태로 코드를 작성해야 한다.
- 이와 같이 작성하였을 때 b_gpu의 경우 2에서 5로 b_value와 값이 교환된 것을 알 수 있다.