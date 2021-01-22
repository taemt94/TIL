// 20210122 TIL
# include <iostream>
# include<cuda_runtime.h>


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