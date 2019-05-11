#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <cuda.h>
#include <darknet.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <driver_types.h>
#include <time.h>
#include "utils.h"
#include "blas.h"

//#include <cutil_inline.h>
#include <cblas.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>


#define BLOCK 512 

static cudaStream_t streamsArray[16];
static int streamInit[16] = { 0 };

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(EXIT_FAILURE);
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}


int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}



cudaStream_t get_cuda_stream() {
    int i = cuda_get_device();
    if (!streamInit[i]) {
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char *s = cudaGetErrorString(status);
            char buffer[256];
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);
            check_error(status);
        }
        streamInit[i] = 1;
    }
    return streamsArray[i];
}


cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        cublasStatus_t status = cublasSetStream(handle[i], get_cuda_stream());
        init[i] = 1;
    }
    return handle[i];
}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //data_im[(channel_in * height + h_in) * width + w_in + i * width + j];
                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_ongpu(const float *im,
                   const int Map[], const int Kernel[],
                   const int Pad[], const int Stride[],
                   const int Channel, 
                   float *data_col){
    int height_col = (Map[0] - Kernel[0] + 2 * Pad[0]) / Stride[0] + 1;
    int width_col = (Map[1] - Kernel[1] + 2 * Pad[1] / Stride[0] + 1);

    int num_kernels = Channel * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels + BLOCK - 1) / BLOCK,
        BLOCK, 0, get_cuda_stream()>>>(num_kernels, im, Map[0], Map[1], Kernel[0], 
        Pad[0], Stride[0], height_col, width_col, data_col);

}

//void simple_sgemm(const float *A, const float *B, float *C) {
//    int i, j, k;
//    for(i=0; i<N; i++)
//    for(j=0; j<N; j++) {
//        float s=0;
//        for(k=0; k<N; k++) s+=A[k*N+i]*B[j*N+k];
//        C[j*N+i]=s;
//    }
//}

int gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    //cudaError_t stream_status = cublasSetStream(handle, get_cuda_stream());
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    //check_error(status);
    if(status != 0){
        printf("CUDA Error: status = %d",status);
        return status;
    }
   
}

__global__ void add_bias_kernel(float *output, float biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] += biases;
}

void add_bias_gpu(float *output, float biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_kernel<<<dimGrid, dimBlock, 0, get_cuda_stream()>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
/* print_2D: 打印2D矩阵 */
void print_2D(const char Name[], const float *A, const int Map[]) {
	
	printf("The %s is:", *Name);

	for (int i = 0; i < Map[1]; i++) {
		for (int j = 0; j < Map[0]; j++) {
			printf("%5.2f%c", A[i * Map[0] + j], (j == Map[0] - 1)?'\n':' ');
        }
    }
    printf("\n");
    return;
}
/* print_3D: 打印3D矩阵 */
void print_3D(const char Name[], const float *A, const int Map[], const int Channel) {
	
	printf("The %s is:", *Name);

	for (int c = 0; c < Channel; c++) {
        for (int i = 0; i < Map[1]; i++) {
            for (int j = 0; j < Map[0]; j++) {
                printf("%5.2f%c", A[Map[1] * Map[0] * c + i * Map[0] + j], (j == Map[0] - 1)?'\n':' ');
                if ((i == Map[1] - 1) && (j == Map[0] - 1)) {
                    printf("\n");
                }
            }
        }
    }
    return;
}

/* convert_3D: 将3D的feature Map 转化为2D Matrix */
void convert_3D(const float *A, float *A_convert, int Channel, const int Map[], const int Kernel[]) {
	/* 计算卷积输出矩阵宽高 */
	const int OutM[2] = {Map[0] - Kernel[0] + 1, Map[1] - Kernel[1] + 1};

	/* 计算(忽略channel的情况下)被卷积矩阵宽高 */
	const int ConvAw = Kernel[0] * Kernel[1];
	// const int ConvAh = OutM[0] * OutM[1];

	/* 给转换后的矩阵A_convert分配空间 */
	/* float A_convert[convAh * convAw * channel] = {0}; */

	int seg = Channel * ConvAw;
	
	for (int c = 0; c < Channel; c++) {
		for (int i = 0; i < OutM[1]; i++) {
			for (int j = 0; j < OutM[0]; j++) {
				int wh = i * seg * OutM[0] + j * seg + c * ConvAw;
				for (int index1 = 0; index1 < Kernel[1]; index1++) {
					for (int index2 = 0; index2 < Kernel[0]; index2++) {
						int col_index = c * Map[0] * Map[1] + (i + index1) * Map[0] + j;
						A_convert[wh + index1 * Kernel[0] + index2] = A[col_index + index2];
					}
				}
			}
		}		
	}
    return;
}

/* matrix_Multiply: 矩阵相乘 */
void matrix_Multiply(const float *A_convert, const float *B, float *C, const int ConvAh, const int ConvAw, const float Bias, const int Channel) {

    /* C := alpha*op(A)*op(B) + beta*C
     * 其中: op(A)表示A矩阵或是转置等. 
     * A是一个M*K矩阵 
     * B是一个K*N矩阵 
     * C是一个M*N矩阵 */

    const enum CBLAS_ORDER Order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
    const int M = ConvAh; // A的行数, C的行数
    const int N = 1;      // B的列数, C的列数
    const int K = ConvAw * Channel; // A的列数, B的行数
    const float alpha = 1;
    const float beta = 0;
    const int lda = K; // A的列
    const int ldb = N; // B的列
    const int ldc = N; // C的列 
    
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A_convert, lda, B, ldb, beta, C, ldc);
    for (int i = 0; i < M * ldc; i++) {
        C[i] = C[i] + Bias;
    }
    return;
}

/* matrix_Pad: 给原始的feature map 加上pad */
void matrix_Pad(const float *A, float *A_pad, const int Channel, const int Map[], const int Pad[]) {
    const int Out_Size[] = {2 * Pad[0] + Map[0], 2 * Pad[1] + Map[1]};

    for (int c = 0; c < Channel; c++) { // 循环Channel
        for (int i = 0; i < Out_Size[1]; i++) { // 循环H
            for (int j = 0; j < Out_Size[0]; j++) { // 循环W
                int col = (c * Out_Size[0] * Out_Size[1]) + i * Out_Size[0] + j; // 计算A_pad的索引
                
                if ((i >= 0 && i < Pad[1])|| (i >= Out_Size[1] - Pad[1] && i < Out_Size[1])) { // 原始feature map上面和下面补零
                    A_pad[col] = 0;
                }
                else { 
                    if ((j >= 0 && j < Pad[0]) || (j >= Out_Size[0] - Pad[0] && j < Out_Size[0])) { // 原始feature map左面和右面补零
                        A_pad[col] = 0;
                    }
                    else {
                        A_pad[col] = A[(c * Map[0] * Map[1]) + (i - Pad[1]) * Map[0] + j - Pad[0]]; // 其余部分不变
                    }
                }
            }
        }
    }
    return;
}

/* convolution3D_Pad: 带Pad的3D卷积(Stride = 1) */
void convolution3D_Pad(const float *A, const float *B, float *C, 
                       const int Map[], const int Kernel[], const int Pad[],
                       const float Bias, 
                       const int Channel) {
    /* Step1: 给A加上pad */
    std::cout << "<===========Step1: Add pad for A: " << std::endl;
    float *A_pad = NULL;
    // printf("A_pad is %u\n", A_pad);
    std::cout << "Malloc memory for A_pad. " << std::endl;
    // A_pad = (float *) malloc(Channel * (Map[0] + 2 * Pad[0]) * (Map[1] + 2 * Pad[1]) * sizeof(float));
    A_pad = new float[Channel * (Map[0] + 2 * Pad[0]) * (Map[1] + 2 * Pad[1]) * sizeof(float)];

    if (A_pad == NULL) {
        printf("申请内存失败!\n");
        return;
    }
    // printf("A_pad is %u\n", A_pad);
    matrix_Pad(A, A_pad, Channel, Map, Pad);
    // printf("A_pad is %u\n", A_pad);
    int Size[2] = { Map[0] + 2 * Pad[0], Map[1] + 2 * Pad[1] };
    print_3D("P", A_pad, Size, Channel);
    // printf("A_pad is %u\n", A_pad);
    // free(A_pad);
    
    /* Step2: 将A_pad转换为2D矩阵 */
    std::cout << "<===========Step2: Convert A_pad to 2D Matrix: " << std::endl;
    float *A_convert = NULL;
    std::cout << "Malloc memory for A_convert. " << std::endl;
    A_convert = new float[Channel * (Kernel[0] * Kernel[1]) * (Map[0] + 2 * Pad[0] - Kernel[0] + 1) * (Map[1] + 2 * Pad[1] - Kernel[1] + 1)];
    
    if (A_convert == NULL) {
        printf("申请内存失败!\n");
        return;
    }
    
    convert_3D(A_pad, A_convert, Channel, Size, Kernel);
    delete[] A_pad;
    
    Size[0] = Channel * (Kernel[0] * Kernel[1]);
    Size[1] = (Map[0] + 2 * Pad[0] - Kernel[0] + 1) * (Map[1] + 2 * Pad[1] - Kernel[1] + 1);
    print_2D("A_convert", A_convert, Size);

    /* Step3: 将A_convert与B相乘*/
    std::cout << "<===========Step3: Multiply A_convert with B --> C: " << std::endl;
    
    /* 计算被卷积宽高 */
    int ConvAw = Kernel[0] * Kernel[1];
    int ConvAh = (Map[0] + 2 * Pad[0] - Kernel[0] + 1) * (Map[1] + 2 * Pad[1] - Kernel[1] + 1);
    matrix_Multiply(A_convert, B, C, ConvAh, ConvAw, Bias, Channel);
	delete[] A_convert;

    Size[0] = Map[0] + 2 * Pad[0] - Kernel[0] + 1;
    Size[1] = Map[1] + 2 * Pad[1] - Kernel[1] + 1;
    // print_2D("C", C, Size);
    
    return;
}

/* convolution3D_Pad_Stride: 计算带Pad和Stride的3D卷积 */
void convolution3D_Pad_Stride(const float *A, const float *B, float *C, 
                              const int Map[], const int Kernel[], 
                              const int Pad[], const int Stride[],
                              const float Bias, 
                              const int Channel) {
	int C_Full_Size[] = {(Map[0] - Kernel[0] + 2 * Pad[0] + 1), (Map[1] - Kernel[1] + 2 * Pad[1] + 1)}; // Stride = 1得到的卷积结果
	float *C_Full = new float[C_Full_Size[1] * C_Full_Size[0]];
    if (C_Full == NULL) {
        printf("分配内存失败!\n");
    }

	convolution3D_Pad(A, B, C_Full, Map, Kernel, Pad, Bias, Channel);
	print_2D("C_Full", C_Full, C_Full_Size);

	int flag = 0;
	for (int i = 0; i < C_Full_Size[1]; i += Stride[1]) {
		for (int j = 0; j < C_Full_Size[0]; j+= Stride[0]) {
			C[flag] = C_Full[i * C_Full_Size[0] + j];
			flag++;
		}
	}
	delete[] C_Full;
	std::cout << "<===========Step4: Consider in Stride: " << std::endl;
	
	return;	
}

/* convolution3D: 完整的3D卷积 */
void convolution3D(const float *A, const float *B, float *C,
                   const int Map[], const int Kernel[],
                   const int Pad[], const int Stride[],
                   const float Bias,
                   const int Channel) {
	convolution3D_Pad_Stride(A, B, C, Map, Kernel, Pad, Stride, Bias, Channel);
	return;
}

int main() { 
    const int Channel = 3;
    const int Map[] = {5, 5}; // W, H
    const int Kernel[] = {3, 3}; // W, H
    const int Pad[] = {1, 1}; // W, H
    const int Stride[] = {1, 2}; // W, H
    float Bias = 1.0;

    int height_col = (Map[0] - Kernel[0] + 2 * Pad[0]) / Stride[0] + 1;
    int width_col = (Map[1] - Kernel[1] + 2 * Pad[1] / Stride[0] + 1); 

    float *h_A=(float*)malloc(Map[0] * Map[1] * sizeof(float));
    float *h_B=(float*)malloc(Kernel[0] * Kernel[1] * sizeof(float));
    float *h_C=(float*)malloc(height_col * width_col * sizeof(float));
    float *h_C_ref=(float*)malloc(fileheight_col * width_col * sizeof(float));
    float *d_A = NULL, *d_B = NULL, *d_C = NULL, *data_col = NULL;
    time_t start_gpu, end_gpu, start_cpu, end_cpu;
    //cutCreateTimer(&timer1);
    //cutStartTimer(timer1);
    printf("CUBLAS test running..\n");
    cublasInit();
    for(int i=0; i<Map[0] * Map[1]; i++) {
        h_A[i]=rand()/(float)RAND_MAX;
        print_3D("A", h_A, Map, Channel);
    }
    for(int j=0; j<Kernel[0] * Kernel[1]; j++) {
        h_B[j]=rand()/(float)RAND_MAX;
        print_3D("B", h_B, Kernel, Channel);
    }
    cublasAlloc(Map[0] * Map[1], sizeof(float), (void**)&d_A);
    cublasAlloc(Kernel[0] * Kernel[1], sizeof(float), (void**)&d_B);
    cublasAlloc(height_col * width_col, sizeof(float), (void**)&d_C);
    cublasSetVector(Map[0] * Map[1], sizeof(float), h_A, 1, d_A, 1);
    cublasSetVector(Kernel[0] * Kernel[1], sizeof(float), h_B, 1, d_B, 1);
    float gpu_t, cpu_t, error_norm=0, ref_norm=0;
    int status;
    cudaThreadSynchronize();
    //t0=cutGetTimerValue(timer1);
    start_gpu = time(&start_gpu);

    if(Kernel[0] == 1){
        status = gemm_ongpu(0, 0, 1, height_col * width_col, Kernel[0] * Kernel[1] * Channel, 1., d_B, Kernel[0] * Kernel[1] * Channel, d_A, height_col * width_col, 1., d_C + 1 * height_col * width_col, height_col * width_col);  
        if (status != 0){
            return 1;
        }  
    }
    else {
        im2col_ongpu(d_A, Map, Kernel, Pad, Stride, Channel, data_col);
        status = gemm_ongpu(0, 0, 1, height_col * width_col, Kernel[0] * Kernel[1] * Channel, 1., d_B, Kernel[0] * Kernel[1] * Channel, data_col, height_col * width_col, 1., d_C + 1 * height_col * width_col, height_col * width_col);
        if (status != 0){
            return 1;
        }  

    }
    add_bias_gpu(d_C, Bias, 1, 1, height_col * width_col);
    cudaThreadSynchronize();
    //cublasSgemm('n', 'n', N, N, N, 1.0f, d_A, N, d_B, N, 0.0f, d_C, N);

    end_gpu = time(&end_gpu);
    gpu_t=(end_gpu - start_gpu)/1000.0f;
    cublasGetVector(height_col * width_col, sizeof(float), d_C, 1, h_C, 1);
    int Output_Size[] = {height_col, width_col};
    print_2D("C", h_C, Output_Size);
    start_cpu = time(&start_cpu);
    //simple_sgemm(h_A, h_B, h_C_ref);

    convolution3D(h_A, h_B, h_C_ref, Map, Kernel, Pad, Stride, Bias, Channel);
    end_cpu = time(&end_cpu);
    cpu_t=(end_cpu - start_cpu)/1000.0f;
    print_2D("C_ref", h_C_ref, Output_Size);

    printf("Map={%4d, %4d}, GPU=%.6fs(%.3fGflops), CPU=%.6fs(%.3fGflops)\n", 
        Map[0], Map[1], gpu_t, 1e-9*Map[0]*Map[1]*Kernel[0]*2/gpu_t, cpu_t, 1e-9*Map[0]*Map[1]*Kernel[0]*2/cpu_t);
    for(int k=0; k<height_col * width_col; k++) {
        float diff=h_C_ref[k]-h_C[k];
        error_norm+=diff*diff;
        ref_norm+=h_C_ref[k]*h_C_ref[k];
    }
    printf("Test %s\n", (sqrtf(error_norm/ref_norm)<1E-6) ? "PASSED" : "FAILED");
}
