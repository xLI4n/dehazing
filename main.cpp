//
//  main.cpp
//  何凯明暗通道导向滤波算法仿真
//
//  Created by 徐振轩 on 15/6/4.
//  Copyright (c) 2015年 徐振轩. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <time.h>

#define MAX_GRAY_VALUE 256
#define MIN_GRAY_VALUE 0

using namespace std;
using namespace cv;

void Dark_Channel(Mat& src, Mat& dst, int Block_size);
double A;
double A_Light(Mat& Img);
void dehaze_image(Mat& src, Mat& dst, Mat& T, float Th, float A);
void guidedFilter(Mat& src,Mat& g_Image,Mat& dst,int r,float e);
int otsu(Mat& dst);
int main()
{
    clock_t start,finish;
    double totletime;
    start = clock();
    Mat srcImage = imread("/Users/xuzhe/Documents/实验数据/雾图及处理后/16.bmp");
    Mat Dark_dstImage(srcImage.size(), CV_32FC3);
    srcImage.convertTo(Dark_dstImage, CV_32FC3);
    Mat Darkchannel_Image(Dark_dstImage.size(), CV_32FC1);
    //imshow("srcImg", srcImage); 
    
    int Block_size=5;
    Dark_Channel(Dark_dstImage, Darkchannel_Image, Block_size);
    A_Light(srcImage);
    cout<<"大气光照平均值A="<<A<<endl;
    
    float W=0.8;
    Mat T(Dark_dstImage.size(), CV_32FC1);
    
    T=1-W/A*Darkchannel_Image;
    //imshow("T", T);
    //T*=255;
    imwrite("/Users/xuzhe/Desktop/透射率图.jpg", T);
    //导向滤波
    Mat g_T(T.size(),CV_32FC1);
    Mat gray_Image(srcImage.size(),CV_32FC1);
    cvtColor(srcImage, gray_Image, CV_BGR2GRAY);//使用原图灰度图作为导向图
    guidedFilter(T, gray_Image, g_T, 15, 0.001);
    //输出透射率图
    //imwrite("/Users/xuzhe/Desktop/导向滤波透射率图.jpg", g_T);

    //int thd = otsu(g_T);
    //cout<<"thd="<<thd<<endl;
    float T0=0.1;
    Mat dstImage(Dark_dstImage.size(), CV_32FC3);
    dehaze_image(Dark_dstImage, dstImage,g_T, T0, A);
    //new_darkmethod_dst
    
    //输出结果
    //imwrite("/Users/xuzhe/Desktop/darkmethod_dst3.jpg", dstImage);
    dstImage/=255;
    imshow("dst", dstImage);
    
    finish = clock();
    totletime = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"主程序运行时间为"<<totletime<<"秒"<<endl;
    
    waitKey();
    
}

void Dark_Channel(Mat& src,Mat& dark_dstImage,int Block_size)
{
    clock_t start,finish;
    double totletime;
    start = clock();
    
    Mat Blue(Block_size,Block_size,CV_32FC1);
    Mat Green(Block_size,Block_size,CV_32FC1);
    Mat Red(Block_size,Block_size,CV_32FC1);
    
    int t = 0;
    double Blue_min = 0;
    double Green_min = 0;
    double Red_min = 0;
    double min_value = 0;
    
    t = (Block_size-1)/2;
    
    vector<Mat>channels;
    split(src, channels);
    for (int i = t; i<dark_dstImage.rows-t; ++i) {
        for (int j = t; j<dark_dstImage.cols-t;++j) {
            Blue = channels.at(0)(Range(i-t,i+t+1),Range(j-t,j+t+1));
            Green = channels.at(1)(Range(i-t,i+t+1),Range(j-t,j+t+1));
            Red = channels.at(2)(Range(i-t,i+t+1),Range(j-t,j+t+1));
            
            minMaxLoc(Blue, &Blue_min,NULL,NULL,NULL);
            minMaxLoc(Green, &Green_min,NULL,NULL,NULL);
            minMaxLoc(Red, &Red_min,NULL,NULL,NULL);
            
            min_value = min(Blue_min, Green_min);
            min_value = min(min_value, Red_min);
            
            dark_dstImage.at<float>(i,j) = (float)min_value;
        }
    }
    
    finish = clock();
    totletime = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"暗通道函数运行时间为"<<totletime<<"秒"<<endl;
}

double A_Light(Mat& Img)
{
    int MaxIndex;
    Mat tmp_mean;
    Mat tmp_stds;
    Mat tmp_score;
    Mat tmp_mean2;
    Mat tmp_stds2;
    double MaxScore;
    double Mean[4];
    double Stds[4];
    double Score[4];
    Mat UpperLeft(cvSize(Img.rows/2, Img.cols/2),CV_8UC3);
    Mat UpperRight(cvSize(Img.rows/2+Img.rows%2, Img.cols/2+Img.cols%2),CV_8UC3);
    Mat LowerLeft(cvSize(Img.rows/2, Img.cols/2),CV_8UC3);
    Mat LowerRight(cvSize(Img.rows/2, Img.cols/2),CV_8UC3);
    
    Mat ImageROI = Img(Rect(0,0,Img.cols/2,Img.rows/2));
    ImageROI.copyTo(UpperLeft);
    Mat ImageROI2 = Img(Rect(Img.cols/2+Img.cols%2,0,Img.cols/2,Img.rows/2));
    ImageROI2.copyTo(UpperRight);
    Mat ImageROI3 = Img(Rect(0,Img.rows/2+Img.rows%2,Img.cols/2,Img.rows/2));
    ImageROI3.copyTo(LowerLeft);
    Mat ImageROI4 = Img(Rect(Img.cols/2+Img.cols%2,Img.rows/2+Img.rows%2,Img.cols/2,Img.rows/2));
    ImageROI4.copyTo(LowerRight);
    
    if (Img.rows*Img.cols>200) {
        
        meanStdDev(UpperLeft,tmp_mean,tmp_stds);
        Mean[0] = tmp_mean.at<double>(0,0);
        Stds[0] = tmp_stds.at<double>(0,0);
        Score[0] = Mean[0] - Stds[0];
        MaxScore = Score[0];
        MaxIndex = 0;
        
        meanStdDev(UpperRight, tmp_mean, tmp_stds);
        Mean[1] = tmp_mean.at<double>(0,0);
        Stds[1] = tmp_stds.at<double>(0,0);
        Score[1] = Mean[1] - Stds[1];
        if (Score[1]>MaxScore) {
            MaxScore = Score[1];
            MaxIndex = 1;
        }
        
        meanStdDev(LowerLeft, tmp_mean, tmp_stds);
        Mean[2] = tmp_mean.at<double>(0,0);
        Stds[2] = tmp_stds.at<double>(0,0);
        Score[2] = Mean[2] - Stds[2];
        if (Score[2]>MaxScore) {
            MaxScore = Score[2];
            MaxIndex = 2;
        }
        
        meanStdDev(LowerRight,tmp_mean, tmp_stds);
        Mean[3] = tmp_mean.at<double>(0,0);
        Stds[3] = tmp_stds.at<double>(0,0);
        Score[3] = Mean[3] - Stds[3];
        if (Score[3]>MaxScore) {
            MaxScore = Score[3];
            MaxIndex = 3;
        }
        
        switch (MaxIndex) {
            case 0:
                A_Light(UpperLeft);
                break;
            case 1:
                A_Light(UpperRight);
                break;
            case 2:
                A_Light(LowerLeft);
                break;
            case 3:
                A_Light(LowerRight);
                break;
        }
    }
    else
    {
        meanStdDev(Img, tmp_mean2, tmp_stds2);
        A = tmp_mean2.at<double>(0,0);
        cout<<"A_tmp = "<<A<<endl;
    }
    return A;
}
void guidedFilter(Mat& src,Mat& g_Image,Mat& dst,int r,float e)
{
    clock_t start,finish;
    double totletime;
    start = clock();
    Mat guided(src.size(),CV_32FC3);
    g_Image.copyTo(guided);
    Mat src_32f,guided_32f;
    src.convertTo(src_32f,CV_32F);
    guided.convertTo(guided_32f,CV_32F);
    
    Mat Ip,I2;
    multiply(guided_32f, src_32f, Ip);
    multiply(src_32f, src_32f, I2);
    Mat mean_p,mean_I,mean_Ip,mean_I2;
    Size w_size(2*r+1,2*r+1);
    boxFilter(src_32f, mean_p, CV_32F, w_size);
    boxFilter(guided_32f, mean_I, CV_32F, w_size);
    boxFilter(Ip, mean_Ip, CV_32F, w_size);
    boxFilter(I2, mean_I2, CV_32F, w_size);
    
    Mat covIp = mean_Ip - mean_I.mul(mean_p);
    Mat E_I = mean_I2 - mean_I.mul(mean_I);
    E_I+=e;
    
    Mat a , b;
    divide(covIp, E_I, a);
    b = mean_p - a.mul(mean_I);
    
    Mat mean_a ,mean_b;
    boxFilter(a, mean_a, CV_32F, w_size);
    boxFilter(b, mean_b, CV_32F, w_size);
    
    dst = mean_a.mul(guided_32f)+mean_b;
    
    finish = clock();
    totletime = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"导向滤波函数运行时间为"<<totletime<<"秒"<<endl;
}
void dehaze_image(Mat& src, Mat& dst, Mat& T, float T0, float A)
{
    float value=0.;
    for(int i=0; i<src.rows;++i)
    {
        for(int j=0; j<src.cols;++j)
        {
            value=max(T0, T.at<float>(i,j));
            dst.at<Vec3f>(i,j)[0]=(src.at<Vec3f>(i,j)[0]-A)/value+A;
            dst.at<Vec3f>(i,j)[1]=(src.at<Vec3f>(i,j)[1]-A)/value+A;
            dst.at<Vec3f>(i,j)[2]=(src.at<Vec3f>(i,j)[2]-A)/value+A;
        }
    }
}
