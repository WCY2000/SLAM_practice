
#include <iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/core/core.hpp>
#include<vector>


int main(int argc, char **argv)
{
	cv::Mat img_1 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img_2 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/2.png", cv::IMREAD_GRAYSCALE);

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	cv::Mat outimg1,outimg2;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/ORB_features1.jpg", outimg1);
	cv::drawKeypoints(img_2, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/ORB_features2.jpg", outimg2);


	std::vector<cv::DMatch> matches;
	matcher->match(descriptors_1, descriptors_2, matches);
    std::cout << matches.size();

    cv::Mat img_raw_match;
	//红色连接的是匹配的特征点数，绿色连接的是未匹配的特征点数
	//matchColor – Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) , the color is generated randomly.
	//singlePointColor – Color of single keypoints(circles), which means that keypoints do not have the matches.If singlePointColor == Scalar::all(-1), the color is generated randomly.
	//CV_RGB(0, 255, 0)存储顺序为R-G-B,表示绿色
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_raw_match, cv::Scalar::all(-1), CV_RGB(0, 0, 255 ), cv::Mat(), 2);
	cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/5_4_raw_match.jpg", img_raw_match);

    double max_dist = matches[0].distance;
	for (int m = 0; m < matches.size(); m++){
		if (matches[m].distance>max_dist){
			max_dist = matches[m].distance;
		}	
	}

	std::vector<cv::DMatch> good_matches;
	for (int m = 0; m < matches.size(); m++){
		if (matches[m].distance < 0.6*max_dist){
			good_matches.push_back(matches[m]);
		}
	}
	
	std::vector <cv::KeyPoint> ransac_keypoint1, ransac_keypoint2;
	for (int i = 0; i < good_matches.size(); i++){
		ransac_keypoint1.push_back(keypoints_1[good_matches[i].queryIdx]);
		ransac_keypoint2.push_back(keypoints_2[good_matches[i].trainIdx]);
	}
	//坐标变换
	std::vector <cv::Point2f> p1, p2;
	for (int i = 0; i < good_matches.size(); i++)
	{
		p1.push_back(ransac_keypoint1[i].pt);
		p2.push_back(ransac_keypoint2[i].pt);
	}

	std::vector<uchar> RansacStatus;
	cv::Mat Fundamental = cv::findFundamentalMat(p1, p2, RansacStatus, cv::FM_RANSAC,1,0.99);

	std::vector <cv::KeyPoint> rancsac_result1, rancsac_result2;
	std::vector <cv::DMatch> rancsac_result_matches;
	int index = 0;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			rancsac_result1.push_back(ransac_keypoint1[i]);
			rancsac_result2.push_back(ransac_keypoint2[i]);
			good_matches[i].queryIdx = index;
			good_matches[i].trainIdx = index;
			rancsac_result_matches.push_back(good_matches[i]);
			index++;
		}
	}
	std::cout << "Number of Correspondence" <<rancsac_result_matches.size();
	cv::Mat img_ransac_matches;
	drawMatches(img_1, rancsac_result1, img_2, rancsac_result2, rancsac_result_matches, img_ransac_matches);
	cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t5/5_4_ransac_match.jpg", img_ransac_matches);


	return 0;
}
