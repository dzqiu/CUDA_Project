#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "stdio.h"
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

#define Accuracy 0
typedef  unsigned char eleType;
__global__ void  DilationStep(eleType *k,eleType *j,unsigned int total)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = x + y*blockDim.x*gridDim.x;

    unsigned int width  = blockDim.x*gridDim.x;
    unsigned int heigth = blockDim.y*gridDim.y;

    if(offset > total) return;

    unsigned int left,right,top,bottom;
    left = offset -1;
    right = offset+1;
    if (x==0) left++;
    if (x==width-1) right--;
    top     = offset - width;
    bottom  = offset + width;
    if (y==0)           top += width;
    if (y==heigth-1)    bottom -= width;

    eleType max = j[offset];
    if(j[left]  -   max > Accuracy)     max = j[left];
    if(j[right] -   max > Accuracy)     max = j[right];
    if(j[bottom]-   max > Accuracy)     max = j[bottom];
    if(j[top]   -   max > Accuracy)     max = j[top];
    unsigned int leftbottom,lefttop,righttop,rightbottom;
    leftbottom  = bottom - 1;
    if(x==0) leftbottom++;
    rightbottom = bottom + 1;
    if(x==width-1) rightbottom--;
    lefttop     = top    - 1;
    if(x==0)    lefttop++;
    righttop    = top    + 1;
    if(x==width-1) righttop--;

    if(j[leftbottom]    -   max > Accuracy)     max = j[leftbottom];
    if(j[rightbottom]   -   max > Accuracy)     max = j[rightbottom];
    if(j[lefttop]       -   max > Accuracy)     max = j[lefttop];
    if(j[righttop]      -   max > Accuracy)     max = j[righttop];

    k[offset] =max;
}
__global__ void PointwiseMinimum(eleType *I,eleType *J,eleType *K,unsigned int total)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = x + y*blockDim.x*gridDim.x;
    if(I[offset] - K[offset] <Accuracy)
        J[offset] = I[offset];
    else
        J[offset] = K[offset];
}

#define DIM 16
Mat imregionmax(const Mat *src,eleType h)
{
     Mat LocMax     = src->clone();
     int width      = src->cols;
     int height     = src->rows;
     Mat Imask      = src->clone();
     Mat Jmasker    = Imask - h;
     Mat K          = Jmasker.clone();
     Mat Tmp        = src->clone();

     eleType *Jmasker_dev,*Imask_dev,*K_dev;

     cudaMalloc((void**)&Jmasker_dev,width*height*sizeof(eleType));
     cudaMemcpy(Jmasker_dev,Jmasker.data,width*height*sizeof(eleType),cudaMemcpyHostToDevice);
     cudaMalloc((void**)&Imask_dev,width*height*sizeof(eleType));
     cudaMemcpy(Imask_dev,Imask.data,width*height*sizeof(eleType),cudaMemcpyHostToDevice);
     cudaMalloc((void**)&K_dev,width*height*sizeof(eleType));

     cudaError_t cudaStatus;
     while(1)
     {

         dim3 blocks(64,64);
         dim3 threads((blocks.x+width-1)/blocks.x,(blocks.y+height-1)/blocks.y);
         DilationStep<<<blocks,threads>>>(K_dev,Jmasker_dev,width*height);
         cudaDeviceSynchronize();
         cudaStatus= cudaGetLastError();
         if (cudaStatus != cudaSuccess)
         {
             fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
             return LocMax;
         }
         cudaMemcpy(K.data,K_dev,width*height*sizeof(eleType),cudaMemcpyDeviceToHost);
         PointwiseMinimum<<<blocks,threads>>>(Imask_dev,Jmasker_dev,K_dev,width*height);
         if (cudaStatus != cudaSuccess)
         {
             fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
             return LocMax;
         }
         cudaMemcpy(Tmp.data,Jmasker_dev,width*height*sizeof(eleType),cudaMemcpyDeviceToHost);
         if (memcmp(Tmp.data,Jmasker.data,width*height*sizeof(eleType))==0) break;
         else cudaMemcpy(Jmasker.data,Jmasker_dev,width*height*sizeof(eleType),cudaMemcpyDeviceToHost);
     }
     cudaFree(Imask_dev);
     cudaFree(Jmasker_dev);
     cudaFree(K_dev);
     LocMax = (Imask-Jmasker>0);
     return LocMax;
}

