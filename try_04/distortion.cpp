#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void distortImg(const Mat &srcImg, Mat &dstImg, const float fx, const float fy, const float cx, const float cy)
{
    int imgHeight=srcImg.rows;
    int imgWidth=srcImg.cols;

    cout << "imgHeight: " << imgHeight;
    cout << "imgWidth:  " << imgWidth;
    cout;
    // imgHeight = 366;
    // imgWidth = 487;

    float disK=0.5;
    // Mat_<float> D = (Mat_<float>(5, 1) << 0.3782, 0.9719, 0, 0, -2.9170);
    
    float A[] = {0.3782, 0.9719, 0, 0, -2.9170};
    Mat D = Mat(5, 1, CV_32F, A);

    uchar* pSrcData=(uchar*)srcImg.data;
    uchar* pDstData=(uchar*)dstImg.data;
    for (int j=0; j<imgHeight; j++)
    {
        for (int i=0; i<imgWidth; i++)
        {
            //转到摄像机坐标系
            float X=(i-cx)/fx;
            float Y=(j-cy)/fy;
            float r2=X*X+Y*Y;
            //加上畸变
            // float newX=X*(1+disK*r2);
            // float newY=Y*(1+disK*r2);
            // cout << D.at<float>(0) << ' ' << D.at<float>(1);
            float newX = X * (1 + D.at<float>(0)*r2 + D.at<float>(1)*r2*r2);
            float newY = Y * (1 + D.at<float>(0)*r2 + D.at<float>(1)*r2*r2);
            //再转到图像坐标系
            float u=newX*fx+cx;
            float v=newY*fy+cy;
            //双线性插值
            int u0=floor(u);
            int v0=floor(v);
            int u1=u0+1;
            int v1=v0+1;

            float dx=u-u0;
            float dy=v-v0;
            float weight1=(1-dx)*(1-dy);
            float weight2=dx*(1-dy);
            float weight3=(1-dx)*dy;
            float weight4=dx*dy;

            int resultIdx=j*imgWidth*3+i*3;
            
            if (u0>=0 && u1<imgWidth && v0>=0 && v1<imgHeight)
            {
                 pDstData[resultIdx+0]=weight1*pSrcData[v0*imgWidth*3 + u0*3+0]+weight2*pSrcData[v0*imgWidth*3+u1*3+0]+weight3*pSrcData[v1*imgWidth*3 +u0*3+0]+weight4*pSrcData[v1*imgWidth*3 + u1*3+0];
                 pDstData[resultIdx+1]=weight1*pSrcData[v0*imgWidth*3 + u0*3+1]+weight2*pSrcData[v0*imgWidth*3+u1*3+1]+weight3*pSrcData[v1*imgWidth*3 +u0*3+1]+weight4*pSrcData[v1*imgWidth*3 + u1*3+1];
                 pDstData[resultIdx+2]=weight1*pSrcData[v0*imgWidth*3 + u0*3+2]+weight2*pSrcData[v0*imgWidth*3+u1*3+2]+weight3*pSrcData[v1*imgWidth*3 +u0*3+2]+weight4*pSrcData[v1*imgWidth*3 + u1*3+2];
            }
        }
    }
}

int main()
{
    string imgPath="/home/wenxiangyu/project/camera_distortion/";
    Mat srcImg = imread(imgPath+"normal_7.jpg");
    // pyrDown(srcImg, srcImg);
    // pyrDown(srcImg, srcImg);

    Mat dstImg = srcImg.clone();
    dstImg.setTo(0);

    namedWindow("showImg",0);
    imshow("showImg", srcImg);
    waitKey(0);

    // float fx=982.140;
    // float fy=982.140;
    // float cx=639.500;
    // float cy=359.500;
    float rate_ = 0.6;

    float fx=982.140;
    float fy=982.140;
    fx *= rate_;
    fy *= rate_;
    float cx=243.500;
    float cy=183.000;
    distortImg(srcImg, dstImg, fx, fy, cx, cy);

    imshow("showImg", dstImg);
    waitKey(0);
    return 0;
}