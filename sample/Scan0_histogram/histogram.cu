#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "stdio.h"
#include <vector>
using namespace std;
using namespace cv;
#define IMAGE_DIR "/home/dzqiu/Documents/image/chaozhou.JPG"

__global__ void  histogram_init(char *src,unsigned int *out)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockDim.y;
    int offset = x + y * gridDim.x * blockDim.x;

    unsigned char value = src[offset];
    atomicAdd(&out[value],1);
}

//refer: Parallel Prefix Sum (Scan) with CUDA, Mark Harris, April 2007
//link : http: //developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf
__global__ void scan_downsweep(unsigned int *in,unsigned int *out,int n)
{
    int Idx = threadIdx.x ;
     extern __shared__ float sdata[];
    sdata[Idx] = in[Idx];
    __syncthreads();

    unsigned int offset=1;

    for(unsigned int i=n>>1;i>0;i>>=1)  //build sum iin place up the tree
    {
        __syncthreads();
        if(Idx<i)
        {
            int ai = offset*(2*Idx+1)-1;
            int bi = offset*(2*Idx+2)-1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2 ;
    }

    if(Idx==0) sdata[n-1]=0;            //travers down tree &build scan
    for(unsigned int i=1;i<n;i<<=1)
    {
        offset >>= 1;
        __syncthreads();
        if(Idx<i)
        {
            int ai = offset*(2*Idx+1)-1;
            int bi = offset*(2*Idx+2)-1;

            int tmp    =sdata[ai];
            sdata[ai]  =   sdata[bi];
            sdata[bi] +=   tmp;
        }
    }
    __syncthreads();
    //out[Idx]=sdata[Idx];
    out[Idx]=((float)sdata[Idx]/512/512*256)-1;
}
__global__ void equal_remap(char *img,char *out,unsigned int *idetity)
{
    int x = threadIdx.x+blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockIdx.y*blockDim.y;
    int offset = x+y*blockDim.x*gridDim.x;

    unsigned char value = img[offset];

    out[offset] = (unsigned char) idetity[value];


}

#define DIM             512
#define GRAY_IDENTITY   256
int main(int argc,char** argv)
{

    Mat img_src = imread(IMAGE_DIR);
    Mat img_gray;
    cvtColor(img_src,img_gray,CV_RGB2GRAY);
    resize(img_gray,img_gray,Size(DIM,DIM));

    unsigned int *dev_histogram;
    cudaMalloc((void**)&dev_histogram,GRAY_IDENTITY*sizeof(int));
    cudaMemset(dev_histogram,0,GRAY_IDENTITY*sizeof(int));

    /*build the histogram of the image*/
    char *dev_gray;
    cudaMalloc((void**)&dev_gray,DIM*DIM);
    cudaMemcpy(dev_gray,img_gray.data,DIM*DIM,cudaMemcpyHostToDevice);
    dim3 blocks(DIM/16,DIM/16);dim3 threads(16,16);
    histogram_init<<<blocks,threads>>>(dev_gray,dev_histogram);
    int host_histogram[GRAY_IDENTITY]={0};
    cudaMemcpy(host_histogram,dev_histogram,GRAY_IDENTITY*sizeof(int),cudaMemcpyDeviceToHost);

    /*equalize the histogram*/
    unsigned int *dev_histogram_equal;
    cudaMalloc((void**)&dev_histogram_equal,GRAY_IDENTITY*sizeof(int));
    blocks=dim3(1,1);threads=dim3(GRAY_IDENTITY,1);
    scan_downsweep<<<blocks,threads,sizeof(int)*GRAY_IDENTITY>>>(dev_histogram,dev_histogram_equal,GRAY_IDENTITY);
    int host_histogram_equal[GRAY_IDENTITY]={0};
    cudaMemcpy(host_histogram_equal,dev_histogram_equal,GRAY_IDENTITY*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<GRAY_IDENTITY/4;i++)
    {
       //if(host_histogram[i]!=0)
         printf("bin[%3d]:%6d -> %6d| bin[%3d]:%6d -> %6d| bin[%3d]:%6d -> %6d| bin[%3d]:%6d -> %6d\n",
                4*i,host_histogram[4*i],host_histogram_equal[4*i],
                4*i+1,host_histogram[4*i+1],host_histogram_equal[4*i+1],
                4*i+2,host_histogram[4*i+2],host_histogram_equal[4*i+2],
                4*i+3,host_histogram[4*i+3],host_histogram_equal[4*i+3]);
    }

    /*remap the image use the equalized histogram*/
    char *dev_equalImg;
    cudaMalloc((void**)&dev_equalImg,DIM*DIM);
    blocks=dim3(DIM/16,DIM/16);threads=dim3(16,16);
    equal_remap<<<blocks,threads>>>(dev_gray,dev_equalImg,dev_histogram_equal);
    Mat equalImg(DIM,DIM,CV_8UC1,Scalar(255));
    cudaMemcpy(equalImg.data,dev_equalImg,DIM*DIM,cudaMemcpyDeviceToHost);




    cudaFree(dev_gray);
    cudaFree(dev_histogram);
    cudaFree(dev_histogram_equal);
    imshow("gray_img",img_gray);
    imshow("histogram equalization use GPU",equalImg);

    waitKey(0);
    return 0;
}
