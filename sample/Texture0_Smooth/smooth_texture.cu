#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "stdio.h"
using namespace std;
using namespace cv;

texture<uchar4,cudaTextureType2D,cudaReadModeNormalizedFloat> tex;
//if using cudaReadModeElementType , cannot read the data to float4.

__global__ void smooth_kernel(char *img,int width,int heigth,int channels)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int offset = x + y*blockDim.x+gridDim.x;

    float u = x/(float)width;
    float v = y/(float)heigth;

    //make sure : tex's cudaTextureReadMode is cudaReadModeNormalizedFloat
    float4 pixel    = tex2D(tex,x,y);
    float4 left     = tex2D(tex,x-1,y);
    float4 right    = tex2D(tex,x+1,y);
    float4 top      = tex2D(tex,x,y-1);
    float4 botton   = tex2D(tex,x,y+1);

    img[(y*width+x)*channels+0] = (left.x+right.x+top.x+botton.x)/4*255;
    img[(y*width+x)*channels+1] = (left.y+right.y+top.y+botton.y)/4*255;
    img[(y*width+x)*channels+2] = (left.z+right.z+top.z+botton.z)/4*255;
    img[(y*width+x)*channels+3] = 0;
}


#define IMAGE_DIR "/home/dzqiu/Documents/zuyan.jpeg"
int main(int argc,char **argv)
{
        Mat src = imread(IMAGE_DIR,IMREAD_COLOR);
        resize(src, src, Size(256, 256));

        //In order to using float texture.
        cvtColor(src, src, CV_BGR2BGRA);

        int rows=src.rows;
        int cols=src.cols;
        int channels=src.channels();
        int width=cols,height=rows,size=rows*cols*channels;

        cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<uchar4>();
        cudaArray *cuArray;
        cudaMallocArray(&cuArray,&channelDesc,width,height);
        cudaMemcpyToArray(cuArray,0,0,src.data,size,cudaMemcpyHostToDevice);

        tex.addressMode[0]=cudaAddressModeWrap;
        tex.addressMode[1]=cudaAddressModeWrap;
        tex.filterMode = cudaFilterModeLinear;
        tex.normalized =false;

        cudaBindTextureToArray(tex,cuArray,channelDesc);


        Mat out=Mat::zeros(width, height, CV_8UC4);
        char *dev_out=NULL;
        cudaMalloc((void**)&dev_out, size);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        smooth_kernel<<<dimGrid,dimBlock,0>>>(dev_out,width,height,channels);

        cudaMemcpy(out.data,dev_out,size,cudaMemcpyDeviceToHost);

        imshow("orignal",src);
        imshow("smooth_image",out);
        waitKey(0);

        cudaFree(dev_out);
        cudaFree(cuArray);
        cudaUnbindTexture(tex);
        return 0;

}
