#include <iostream>
#include <opencv2/opencv.hpp>


int main()
{
    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 458.654, 0.0,367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 );


    cv::Mat src = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/1.png");
    cv::cvtColor(src, src, CV_RGB2GRAY);

    //using initUndistortRectifyMap + remap
    cv::Mat map11, map12;
    cv::Mat map21, map22;
    const double alpha1 = 1;
    const double alpha2 = 0;
    cv::Mat dst1,dst2;

    cv::Mat NewCameraMatrix1 = getOptimalNewCameraMatrix(K, D, src.size(), alpha1, src.size(), 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix1, src.size(), CV_16SC2, map11, map12);
     remap(src, dst1, map11, map12, cv::INTER_LINEAR); 

    cv::Mat NewCameraMatrix2 = getOptimalNewCameraMatrix(K, D, src.size(), alpha2, src.size(), 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix2, src.size(), CV_16SC2, map21, map22);
     remap(src, dst2, map21, map22, cv::INTER_LINEAR); 
     
    cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/5_3_alpha=1.png", dst1);
    cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/5_3_alpha=0.png", dst2);

    //using undistort
    cv::Mat dst3;
    cv::undistort(src, dst3, K, D, K);
    cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/undistort.png", dst2);

    return 0;
}