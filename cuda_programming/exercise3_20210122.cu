// 20210122 TIL
# include <iostream>
# include<cuda_runtime.h>

// 2차원 행렬 곱을 수행하는 예제이다.
// GPU와 CPU 상에서 같은 연산을 수행한 후 수행 시간을 비교한다.
// GPU 상에서 연산을 수행할 때에는 메모리 CPU -> GPU 복사, 커널 호출 시 동기화, 
// GPU -> CPU 복사 등 CPU 상에서 하는 연산 대비 추가되는 과정에 의한 오버헤드가 존재하므로, 
// 연산 반복 횟수가 이러한 오버헤드를 감수할 정도로 충분히 많은지 확인해보아야 한다.
// 아래의 예제는 단순한 곱셈 연산을 1회만 수행하므로 CPU 상에서 연산하는 것이 오히려 GPU 상에서 연산하는 것보다
// 빠르게 측정되지만, 같은 연산을 여러번 반복할 경우 GPU의 오버헤드까지 포함한 연산 속도가 CPU의 연산 속도보다
// 빨라지는 시점이 존재한다.
// 실제 프로그래밍을 할 때에는 이러한 것들을 고려하여 CPU와 GPU 중 어느 것이 효율적인지 직접 테스트해보는 것이 좋다.

__global__ void MatrixMul(int* M, int* N, int* P, int Width)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = Width * ty + tx;

    int Value = 0;
    int Mval = 0;
    int Nval = 0;

    for (int i = 0; i < Width; i++)
    {
        Mval = M[ty * Width + i];
        Nval = N[i * Width + tx];
        Value += Mval * Nval;
    }

    P[tid] = Value;
}
void MatrixMulC(int* M, int* N, int* P, int Width)
{
    int col = 0;
    int raw = 0;
    int index = 0;
    int Destindex = 0;
    for (col = 0; col < Width; col++)
    {
        for (raw = 0; raw < Width; raw++)
        {
            Destindex = col * Width + raw;
            for (index = 0; index < Width; index++)
            {
                //std::cout << M[col * Width + index] << " " << N[index * Width + raw] << std::endl;
                P[Destindex] += M[col * Width + index] * N[index * Width + raw];
            }
        }
    }
}
int main()
{
    const int MatrixWidth = 2;
    const int MatrixHeight = 2;
    const int MatrixSize = MatrixWidth * MatrixHeight;
    const int BufferSize = MatrixSize * sizeof(int);

    int* M;
    int* N;
    int* P_cuda;
    int* P_C;

    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_cuda = (int*)malloc(BufferSize);
    P_C = (int*)malloc(BufferSize);

    int i = 0;

    for (int i = 0; i < MatrixSize; i++)
    {
        M[i] = i;
        N[i] = i;
        P_cuda[i] = 0;
        P_C[i] = 0;
    }
    /*
    for (int i = 0; i < MatrixSize; i++)
    {
        std::cout << M[i] << N[i] << P_cuda[i] << P_C[i] << std::endl;
    }
    */
    int* dev_M;
    int* dev_N;
    int* dev_P;

    cudaMalloc((void**)&dev_M, BufferSize);
    cudaMalloc((void**)&dev_N, BufferSize);
    cudaMalloc((void**)&dev_P, BufferSize);

    cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);

    dim3 Dg(2, 1, 1);
    dim3 Db(2, 1, 1);

    MatrixMul << <Dg, Db >> > (dev_M, dev_N, dev_P, MatrixWidth);
    cudaMemcpy(P_cuda, dev_P, BufferSize, cudaMemcpyDeviceToHost);

    MatrixMulC(M, N, P_C, MatrixWidth);

    bool ResultFlag = true;
    for (int i = 0; i < MatrixSize; i++)
        printf("P_C[%d]: %d\n", i, P_C[i]);

    for (i = 0; i < MatrixSize; i++)
    {
        //std::cout << "CUDA : " << P_cuda[i] << " C : " << P_C[i] << std::endl;

        if (P_cuda[i] != P_C[i])
        {
            ResultFlag = false;
            std::cout << "CUDA : " << P_cuda[i] << " C : " << P_C[i] << std::endl;
        }
    }

    if (ResultFlag == true) std::cout << "MatixMul Result Ok" << std::endl;
    else std::cout << "MatixMul Result Error" << std::endl;

    cudaFree(dev_M);
    cudaFree(dev_N);
    cudaFree(dev_P);

    free(M);
    free(N);
    free(P_cuda);
    free(P_C);

    return 0;
}