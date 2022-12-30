
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// nvcc -arch=compute_35 -Wno-deprecated-gpu-targets addVector.cu 



#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void matvec_mul(float *a, float *v, float * r, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
 
    if (i < n){
        float sum = 0.;
        for( int j = 0; j < n; j++ ) {
            sum += a[i*n + j] * v[j];
        }
        v[i] = sum;
    } 
}

__global__ void mat_mul(float *a, float *b, float * r, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
 
    if (i < n && j < n){
        float sum = 0.;
        for( int k = 0; k < n; k++ ) {
            sum += a[i*n + k] * b[k*n + j];
        }
        r[i*n + j] = sum;
    }
}

#define BLOCK_SIZE 32

__global__ void mat_mul_shared(float *a, float *b, float * r, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    int i_local = threadIdx.x;
    int j_local = threadIdx.y;
    
    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0;
    
    for( int k = 0; k < n / BLOCK_SIZE; k++ ) {
    
        a_tile[i_local][j_local] = a[i * n + k*BLOCK_SIZE + j_local  ];
        b_tile[i_local][j_local] = b[(k*BLOCK_SIZE + i_local)*n +   j];
        
        __syncthreads();    
        for( int k = 0; k < BLOCK_SIZE; k++ ) {
            sum += a_tile[i_local][k] * b_tile[k][j_local];
        }
        __syncthreads();
    
    }
    
    r[i*n + j] = sum;    
}


int main() {
    int  n = 16384;
    dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numberOfBlocks( n / BLOCK_SIZE, n / BLOCK_SIZE );
    
    float *h_a;
    float *h_b;
    float *h_r;
    
   
    float *d_a;
    float *d_b;
    float *d_r;
    
    
    
    size_t size = n*n*sizeof(float);
    
    
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_r = (float*)malloc(size);
    
    for( int i = 0; i < n; i++ ) {
        for( int j = 0; j < n; j++ ) {
            h_a[i*n + j] = h_b[i*n + j] = 1.0;  
        }
    }   
    
    cudaEvent_t start, stop;
    float elapsedTime;

    // Allocate memory for each vector on GPU
    GCE( cudaMalloc(&d_a, size) );
    GCE( cudaMalloc(&d_b, size) );
    GCE( cudaMalloc(&d_r, size) );
    
    GCE( cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice));
    GCE( cudaMemcpy( d_b, h_b, size, cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    
    //mat_mul_shared<<<blocksize, numberOfBlocks >>>(d_a, d_b, d_r, n );
    mat_mul<<< blocksize, numberOfBlocks>>>(d_a, d_b, d_r, n );
    
    GCE( cudaMemcpy( h_r, d_r, size, cudaMemcpyDeviceToHost ) );

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);    
     
    
   // for( int i = 0; i < n; i++ ) {
   //     for( int j = 0; j < n; j++ ) {
   //         printf("%.2lf\t", h_r[i*n + j]);  
   //     }
   //     printf("\n");
   // }

    return 0;
}


