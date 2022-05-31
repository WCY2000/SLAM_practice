#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


int main()    
{                                  

    cv::Mat src = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/1.png");
    int height = src.rows;
	int width = src.cols;

	// for (int row = 0; row < src.rows; row++) {
	// 	uchar* p1 = src.ptr<uchar>(row);
	// 	uchar* p2 = dst.ptr<uchar>(row);
	// 	for (int col = 0; col < src.cols ; col++) {
	// 		p2[col] = 255-p1[col];
	// 	}
	// }
    cv::cvtColor(src, src, CV_RGB2GRAY);
    cv::Mat dst = 255 - src;

    cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/5_2_output.jpg", dst);


    return 0;
}