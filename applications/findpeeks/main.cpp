#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
typedef  unsigned char eleType;
Mat imregionmax(const Mat *src,eleType h);

void LabShow(char wname[20],Mat src,Mat Label,Size offset)
{

    Mat sImg = src.clone();
    if(sImg.channels()<3)
        cvtColor(sImg,sImg,COLOR_GRAY2RGB);
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 1;
    for(int x=0;x<Label.cols;x++)
    {
        for(int y=0;y<Label.cols;y++)
        {
            Point pos;
            pos.x=x+offset.width;
            pos.y=y+offset.height;
            if(Label.at<uchar>(y,x))
                putText(sImg,"x",pos,font_face,font_scale,Scalar(0,0,255),3,8);
        }
    }
    imshow(wname,sImg);
}


int main(int argc, char *argv[])
{

    Mat src(Size(1024,1024),CV_8UC1);
    src.setTo(0);
    circle(src,Point(200,200),100,Scalar(255),-1);
    circle(src,Point(300,300),100,Scalar(255),-1);
    circle(src,Point(400,600),100,Scalar(255),-1);
    circle(src,Point(550,600),100,Scalar(255),-1);
    namedWindow("input", CV_WINDOW_NORMAL);
    imshow("input",src);

    Mat dis=Mat(src.size(),CV_32FC1);
    distanceTransform(src,dis,CV_DIST_L2,3);
    normalize(dis,dis,1.0,0.0,NORM_MINMAX);
    dis = dis *255;dis.convertTo(dis,CV_8U);
    namedWindow("distance Transfrom", CV_WINDOW_NORMAL);
    imshow("distance Transfrom",dis);

    namedWindow("peeks", CV_WINDOW_NORMAL);
    Mat peeks = imregionmax(&dis,1);
    LabShow("peeks",dis,peeks,Size(0,0));

    waitKey(0);
    return 0;
}
