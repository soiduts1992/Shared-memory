#include "bitmap.h"
#include <stdlib.h>
#include <math.h>


#define GRID_SIZE 16
#define BLOCK_SIZE  4 
#define GCE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void transform_to_bw( unsigned char * in, unsigned char * out, int size ) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;


    if(i < size) 
        out[i]  = (unsigned char) round( 0.11 * in[ 3*i + 0 ] + 
                                         0.59 * in[ 3*i + 1 ] +
                                         0.39 * in[ 3*i + 2 ] );
        
} 

__global__ void sobel_filter( unsigned char * in, unsigned char * out, int width, int height ) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    int Gx[9] = { -1,  0,  1, -2,  0,  2, -1,  0,  1 };
    int Gy[9] = { -1, -2, -1,  0,  0,  0,  1,  2,  1 };
      
    if(i > 0 && i < height-1) {
        if(j > 0 && j < width-1) {
            int dx = 0, dy = 0; 
            for( int x = 0; x < 3; x++ ) {
                for( int y = 0; y < 3; y++ ) {
                    dx += in[(i+x-1)*width + j + y - 1]*Gx[3*x+y];
                    dy += in[(i+x-1)*width + j + y - 1]*Gy[3*x+y];
                }       
            }
            
            float value = sqrt( (float) dx*dx + dy*dy );
            out[i*width + j] = value >= 0 ? (unsigned char)round(value) : 0;
        }
    }    

}


int main (int argc, char ** argv) {
    
    if( argc != 3 ) {
        fprintf(stderr, "Sobel Filter, use: %s <input bitmap image> <output bitmap image>\n", argv[0] );
        return 1;
    }
    
    BITMAPINFOHEADER header;
    unsigned char * h_img = bmp_load( (char *)argv[1], &header );
    int width = header.biWidth;
    int height = header.biHeight;
    int size = width * height;

    dim3 nthreadsPerBlock( GRID_SIZE,GRID_SIZE,1 );
    dim3 nBlocsPerGrid( ceil(height/GRID_SIZE), ceil(width/GRID_SIZE), 1 ); 
    dim3 blocksize(BLOCK_SIZE,BLOCK_SIZE);
    dim3 numberOfBlocks(ceil(height/BLOCK_SIZE), ceil(width/BLOCK_SIZE));
    
    unsigned char * h_img_sobel = (unsigned char *) malloc(  size );

    cudaEvent_t start, stop;
    float elapsedTime;

    unsigned char * d_img       = (unsigned char *)  malloc(  size*3 );
    unsigned char * d_img_bw    = (unsigned char *) malloc(  size );
    unsigned char * d_img_sobel = (unsigned char *) malloc(  size );
    GCE( cudaMalloc(&d_img, size*3) );
    GCE( cudaMalloc(&d_img_bw, size) );
    GCE( cudaMalloc(&d_img_sobel, size) );

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    GCE( cudaMemcpy( d_img, h_img, size*3, cudaMemcpyHostToDevice));

    transform_to_bw<<< ceil(size/GRID_SIZE),GRID_SIZE >>>( d_img, d_img_bw,  size );
    sobel_filter<<< nBlocsPerGrid,nthreadsPerBlock >>>( d_img_bw, d_img_sobel,  width, height );
        
    GCE( cudaMemcpy( h_img_sobel, d_img_sobel, size, cudaMemcpyDeviceToHost ) );

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);    

    bmp_save_bw((char *)argv[2], width, height, h_img_sobel );
    
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);   
    
    cudaFree(d_img);
    cudaFree(d_img_bw);
    cudaFree(d_img_sobel);
    free( h_img );
    
    return 0;
}




