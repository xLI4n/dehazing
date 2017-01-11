//
//  main.cpp
//  OTSU
//
//  Created by 徐振轩 on 15/7/3.
//  Copyright (c) 2015年 徐振轩. All rights reserved.
//
/*
#include <iostream>

#include <stdio.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#define MAX_GRAY_VALUE 256
#define MIN_GRAY_VALUE 0

using namespace std;
using namespace cv;
   //腐蚀操作
    
    Mat element = getStructuringElement(MORPH_RECT, Size(11,11));
    Mat out;
    erode(dst, out, element);
    imshow("腐蚀操作", out);
    imwrite("/Users/xuzhenxuan/Desktop/腐蚀后.jpg", out);
    //提取MASK划定ROI
    //Mat mask;
    //cvtColor(out, mask, CV_RGB2GRAY);
    //mshow("mask", mask);
    Mat Fin_dst(src.size(),CV_32FC3);
    src.copyTo(Fin_dst, dst);
    //cvCopy(&src, &Fin_dst,&out);
    
    imshow("Finnal", Fin_dst);
    imwrite("/Users/xuzhenxuan/Desktop/A_Value.jpg", Fin_dst);
*/
/*
    finish = clock();
    totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"此程序运行时间为"<<totaltime<<"秒"<<endl;
    cout<<"此图中天空区域像素点数目约为："<<n<<"个"<<endl;
    per = n/(dst.rows*dst.cols)*100;
    cout<<"天空区域占比约为："<<per<<"%"<<endl;
    waitKey(0);
    return 0;
}
*/
/*
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
    
    Mat srcImage = imread("/Users/xuzhenxuan/Documents/数据/雾图及处理后/116.bmp");
    Mat otsu_srcImage;
    cvtColor(srcImage, otsu_srcImage, CV_RGB2GRAY);
    int thd  = otsu(otsu_srcImage);
    cvtColor(otsu_srcImage, otsu_srcImage, CV_GRAY2RGB);
    cout<<"thd = "<<thd<<endl;
    for (int i = 0; i<otsu_srcImage.rows; ++i) {
        for (int j = 0; j<otsu_srcImage.cols*otsu_srcImage.channels(); ++j) {
            if (otsu_srcImage.at<uchar>(i,j)>thd) {
                otsu_srcImage.at<uchar>(i,j)=0;
            } else {
                otsu_srcImage.at<uchar>(i,j)=1;
            }
        }
    }
    
    Mat darkchannelmethod_src;
    multiply(srcImage, otsu_srcImage, darkchannelmethod_src);
    imwrite("/Users/xuzhenxuan/Desktop/dark_src2.jpg", darkchannelmethod_src);
    Mat otsu_srcImage2 = otsu_srcImage.clone();
    for (int i  = 0; i<otsu_srcImage2.rows;++i) {
        for (int j = 0; j<otsu_srcImage2.cols*otsu_srcImage2.channels(); ++j) {
            otsu_srcImage2.at<uchar>(i,j) = 1 - otsu_srcImage2.at<uchar>(i,j);
        }
    }
    
    Mat clahemethod_src;
    multiply(srcImage, otsu_srcImage2, clahemethod_src);
    imwrite("/Users/xuzhenxuan/Desktop/dark_src3.jpg", clahemethod_src);

    
    Mat srcImage = imread("/Users/xuzhenxuan/Documents/数据/雾图及处理后/116.bmp");
    Mat Dark_dstImage(srcImage.size(),CV_32FC3);
    
    srcImage.convertTo(Dark_dstImage, CV_32FC3);
    Mat Darkchannel_Image(Dark_dstImage.size(), CV_32FC1);
    
    int Block_size=5;
    Dark_Channel(Dark_dstImage, Darkchannel_Image, Block_size);
    
    A_Light(srcImage);
    cout<<"大气光照平均值A="<<A<<endl;
    
    float W=0.8;
    Mat T(Dark_dstImage.size(), CV_32FC1);
    T=1-W/A*Darkchannel_Image;
    imshow("T", T);
    
    Mat g_T(T.size(),CV_32FC1);
    Mat gray_Image(srcImage.size(),CV_32FC1);
    cvtColor(srcImage, gray_Image, CV_BGR2GRAY);//使用原图灰度图作为导向图
    guidedFilter(T, gray_Image, g_T, 15, 0.001);
    
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
int otsu(Mat& dst){
    
    int i,j;
    int tmp;
    double u0,u1,w0,w1,u = 0.0, uk;
    double cov;
    double maxcov=0.0;
    int maxthread=0;
    
    int hst[MAX_GRAY_VALUE]={0};
    double pro_hst[MAX_GRAY_VALUE]={0.0};
    //统计每个灰度的数量
    for( i =0 ; i<dst.rows; i++ ){
        for( j=0; j<dst.cols; j++){
            tmp=dst.at<uchar>(i,j);
            hst[tmp]++;
        }
    }
    //计算每个灰度级占图像中的概率
    for( i=MIN_GRAY_VALUE ; i<MAX_GRAY_VALUE; i++)
        pro_hst[i]=(double)hst[i]/(double)(dst.rows*dst.cols);
    //计算平均灰度值
    for( i=MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++)
        u += i*pro_hst[i];
    //计算方差
    double det=0.0;
    for( i= MIN_GRAY_VALUE; i< MAX_GRAY_VALUE; i++)
        det += (i-u)*(i-u)*pro_hst[i];
    //统计前景和背景的平均灰度值，并计算类间方差
    for( i=MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++){
        w0=0.0; w1=0.0; u0=0.0; u1=0.0; uk=0.0;//必须执行此循环初始化。
        for( j=MIN_GRAY_VALUE; j < i; j++){
            uk += j*pro_hst[j];
            w0 += pro_hst[j];
        }
        u0=uk/w0;
        w1=1-w0;
        u1= (u - uk )/(1-w0);
        //计算类间方差
        cov=w0*w1*(u1-u0)*(u1-u0);
        
        if ( cov > maxcov )
        {
            maxcov=cov;
            maxthread=i;
        }
    }
    cout<<maxthread<<endl;
    return maxthread;
}


*/

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
    ///Users/xuzhenxuan/Desktop/新darkmethod_src.jpg
    Mat srcImage = imread("/Users/xuzhe/Desktop/WechatIMG1.jpeg");
    //Mat srcImage_a = imread("/Users/xuzhenxuan/Documents/数据/雾图及处理后/building3.png");
    Mat Dark_dstImage(srcImage.size(), CV_32FC3);
    srcImage.convertTo(Dark_dstImage, CV_32FC3);
    Mat Darkchannel_Image(Dark_dstImage.size(), CV_32FC1);
    //imshow("srcImg", srcImage); 
    
    int Block_size=5;
    Dark_Channel(Dark_dstImage, Darkchannel_Image, Block_size);
    //Mat showDarkchennel(Darkchannel_Image.size(),CV_32FC1);
    //Darkchannel_Image.copyTo(showDarkchennel);
    //showDarkchennel/=255;
    //imshow("Darkchannel", showDarkchennel);
    //imwrite("/Users/xuzhenxuan/Desktop/暗通道图___tu5.jpg", Darkchannel_Image);
    //GaussianBlur(Darkchannel_Image, Darkchannel_Image, Size(11,11), 0,0);
    //Mat element = getStructuringElement(MORPH_RECT, Size(11,11));
    //Mat out;
    //erode(Darkchannel_Image, out, element);
    //GaussianBlur(out, out, Size(11,11),0,0);
    //Darkchannel_Image/=255;
    //imshow("erzhiantongdao", Darkchannel_Image);
    //imwrite("/Users/xuzhenxuan/Desktop/腐蚀后暗通道5.jpg", out);
    //waitKey();
    //Darkchannel_Image*=255;
    imshow("darkchannel", Darkchannel_Image);
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
    imwrite("/Users/xuzhe/Desktop/导向滤波透射率图.jpg", g_T);
    //int thd = otsu(g_T);
    //cout<<"thd="<<thd<<endl;
    float T0=0.1;
    Mat dstImage(Dark_dstImage.size(), CV_32FC3);
    dehaze_image(Dark_dstImage, dstImage,g_T, T0, A);
    //new_darkmethod_dst
    imwrite("/Users/xuzhe/Desktop/darkmethod_dst3.jpg", dstImage);
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
    int i,j;
    int val = i - t;
    int val2 = j - t;
    int val_i = i+t+1;
    int val_j = j+t+1;
    for (i = 0; i<dark_dstImage.rows; ++i) {
        for (j = 0; j<dark_dstImage.cols; ++j) {
            if (val<0) {
                val = 0;
            }
            if (val2<0) {
                val2 = 0;
            }
            if (val_i>dark_dstImage.rows+1) {
                val_i = dark_dstImage.rows+1;
            }
            if (val_j>dark_dstImage.cols+1) {
                val_j = dark_dstImage.cols+1;
            }
            Blue = channels.at(0)(Range(val,val_i),Range(val2,val_j));
            Green = channels.at(1)(Range(val,val_i),Range(val2,val_j));
            Red = channels.at(2)(Range(val,val_i),Range(val2,val_j));
            
            minMaxLoc(Blue, &Blue_min,NULL,NULL,NULL);
            minMaxLoc(Green, &Green_min,NULL,NULL,NULL);
            minMaxLoc(Red, &Red_min,NULL,NULL,NULL);
            
            min_value = min(Blue_min, Green_min);
            min_value = min(min_value, Red_min);
            
            dark_dstImage.at<float>(i,j) = (float)min_value;
        }
    }
    /*
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
     */
    
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
int otsu(Mat& dst)
{
    
    int i,j;
    int tmp;
    double u0,u1,w0,w1,u = 0.0, uk;
    double cov;
    double maxcov=0.0;
    int maxthread=0;
    
    int hst[MAX_GRAY_VALUE]={0};
    double pro_hst[MAX_GRAY_VALUE]={0.0};
    //统计每个灰度的数量
    for( i =0 ; i<dst.rows; i++ ){
        for( j=0; j<dst.cols; j++){
            tmp=dst.at<uchar>(i,j);
            hst[tmp]++;
        }
    }
    //计算每个灰度级占图像中的概率
    for( i=MIN_GRAY_VALUE ; i<MAX_GRAY_VALUE; i++)
        pro_hst[i]=(double)hst[i]/(double)(dst.rows*dst.cols);
    //计算平均灰度值
    for( i=MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++)
        u += i*pro_hst[i];
    //计算方差
    double det=0.0;
    for( i= MIN_GRAY_VALUE; i< MAX_GRAY_VALUE; i++)
        det += (i-u)*(i-u)*pro_hst[i];
    //统计前景和背景的平均灰度值，并计算类间方差
    for( i=MIN_GRAY_VALUE; i<MAX_GRAY_VALUE; i++){
        w0=0.0; w1=0.0; u0=0.0; u1=0.0; uk=0.0;//必须执行此循环初始化。
        for( j=MIN_GRAY_VALUE; j < i; j++){
            uk += j*pro_hst[j];
            w0 += pro_hst[j];
        }
        u0=uk/w0;
        w1=1-w0;
        u1= (u - uk )/(1-w0);
        //计算类间方差
        cov=w0*w1*(u1-u0)*(u1-u0);
        
        if ( cov > maxcov )
        {
            maxcov=cov;
            maxthread=i;
        }
    }
    cout<<maxthread<<endl;
    return maxthread;
}
























