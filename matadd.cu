#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

#define TIME_START gettimeofday(&t_start, NULL);
#define TIME_END(name)                                         \
    gettimeofday(&t_end, NULL);                                \
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;    \
    elapsedTime += (t_end.tv_usec - t_start.tv_usec) / 1000.0; \
    printf(#name " Time = %f ms.\n", elapsedTime);

typedef struct
{
    size_t rows;
    size_t cols;
    float *data;        // CPU memory
    float *data_device; // GPU mememory
} Matrix;

Matrix *createMatrix(size_t r, size_t c)
{
    size_t len = r * c;
    if (len == 0)
    {
        fprintf(stderr, "Invalid size. The input should be > 0.\n");
        return NULL;
    }
    Matrix *p = (Matrix *)malloc(sizeof(Matrix));
    if (p == NULL)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    p->rows = r;
    p->cols = c;
    p->data = (float *)malloc(sizeof(float) * len);
    if (p->data == NULL)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    if (cudaMalloc(&p->data_device, sizeof(float) * len) != cudaSuccess)
    {
        fprintf(stderr, "Allocate device memory failed.\n");
        goto ERR_TAG;
    }
    return p;
ERR_TAG:
    if (p && p->data)
        free(p->data);
    if (p)
        free(p);
    return NULL;
}

void freeMatrix(Matrix **pp)
{
    if (pp == NULL)
        return;
    Matrix *p = *pp;
    if (p != NULL)
    {
        if (p->data)
            free(p->data);
        if (p->data_device)
            cudaFree(p->data_device);
    }
    *pp = NULL;
}
// a simple function to set all elements to the same value
bool setMatrix(Matrix *pMat, float val)
{
    if (pMat == NULL)
    {
        fprintf(stderr, "NULL pointer.\n");
        return false;
    }
    size_t len = pMat->rows * pMat->cols;
    for (size_t i = 0; i < len; i++)
        pMat->data[i] = val;

    return true;
}

bool setMatrixRandomly(Matrix *pmat, int begin, int end)
{
    srand((unsigned int)time_t(NULL));
    size_t len = pmat->cols * pmat->rows;
    for (size_t i = 0; i < len; i++)
    {
        pmat->data[i] = (rand() % (end - begin) + begin) * (1 + (float)(rand() % 2)/10);
    }
    return true;
}

// B = aA + b
bool linearCPU(const Matrix *pmat, float a, float b, Matrix *pResult)
{
    if (pmat == NULL || pResult == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }

    if (pmat->cols != pResult->cols || pmat->rows != pResult->rows)
    {
        fprintf(stderr, "Matrices have different sizes.\n");
        return false;
    }
    size_t len = pmat->rows * pmat->cols;
    for (size_t i = 0; i < len; i++)
    {
        pResult->data[i] = a * pmat->data[i] + b;
    }
    return true;
}

bool addCPU(const Matrix *pMat1, const Matrix *pMat2, Matrix *pMat3)
{
    if (pMat1 == NULL || pMat2 == NULL || pMat3 == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat1->rows != pMat2->rows || pMat1->cols != pMat2->cols ||
        pMat2->rows != pMat3->rows || pMat2->cols != pMat3->cols)
    {
        fprintf(stderr, "The 3 matrics are not in the same size.\n");
        return false;
    }
    size_t len = pMat1->rows * pMat1->cols;
    for (int i = 0; i < len; i++)
        pMat3->data[i] = pMat1->data[i] + pMat2->data[i];
    return true;
}

__global__ void addKernel(const float *input1, const float *input2, float *output, size_t len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        output[i] = input1[i] + input2[i];
}

__global__ void linearKernel(const float *matrix, float a, float b, float *output, size_t len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        output[i] = matrix[i] * a + b;
}

bool linearGPU(const Matrix *pmat, float a, float b, Matrix *pResult)
{
    if (pmat == NULL || pResult == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }

    if (pmat->cols != pResult->cols || pmat->rows != pResult->rows)
    {
        fprintf(stderr, "Matrices have different sizes.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    size_t len = pmat->rows * pmat->cols;

    cudaMemcpy(pmat->data_device, pmat->data, sizeof(float) * len, cudaMemcpyHostToDevice);
    linearKernel<<<(len + 255) / 256, 256>>>(pmat->data_device, a, b, pResult->data_device, len);
    if ((ecode = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(pResult->data, pResult->data_device, sizeof(float) * len, cudaMemcpyDeviceToHost);
    return true;
}

bool addGPU(const Matrix *pMat1, const Matrix *pMat2, Matrix *pMat3)
{
    if (pMat1 == NULL || pMat2 == NULL || pMat3 == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat1->rows != pMat2->rows || pMat1->cols != pMat2->cols ||
        pMat2->rows != pMat3->rows || pMat2->cols != pMat3->cols)
    {
        fprintf(stderr, "The 3 matrics are not in the same size.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    size_t len = pMat1->rows * pMat1->cols;

    cudaMemcpy(pMat1->data_device, pMat1->data, sizeof(float) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(pMat2->data_device, pMat2->data, sizeof(float) * len, cudaMemcpyHostToDevice);
    addKernel<<<(len + 255) / 256, 256>>>(pMat1->data_device, pMat2->data_device, pMat3->data_device, len);
    if ((ecode = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(pMat3->data, pMat3->data_device, sizeof(float) * len, cudaMemcpyDeviceToHost);

    return true;
}

int main()
{

    struct timeval t_start, t_end;
    double elapsedTime = 0;

    int dev_count = 0;
    int dev_id = 0;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(0);
    cudaGetDevice(&dev_id);
    printf("You have %d cuda devices.\n", dev_count);
    printf("You are using device %d.\n", dev_id);

    Matrix *pmat = createMatrix(4096, 4096);
    Matrix *pResult = createMatrix(4096,4096);
    float a = 3.0;
    float b = 2.0;
    setMatrixRandomly(pmat,1,10);

    TIME_START
    linearCPU(pmat,a,b,pResult);
    TIME_END(linearCPU)
    

    TIME_START
    linearGPU(pmat,a,b,pResult);
    TIME_END(linearGPU)


    freeMatrix(&pmat);
    freeMatrix(&pResult);
    return 0;
}