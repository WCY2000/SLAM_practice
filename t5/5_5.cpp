
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <random>
#include <set>

bool varify_R_t(Eigen::Vector4d p1, Eigen::Vector4d p2, Eigen::Matrix<double, 3,3> R,  Eigen::Vector4d t){
    Eigen::Matrix4d T = Eigen::MatrixXd::Zero(4, 4);
    T.block<3, 3>(0, 0) = R;
    T.block<4, 1>(0, 3) = t;
    // std::cout << "line 20:" << T * p1 << std::endl;
    return ( (T * p1)(2) > 0);
}

int main(int argc, char **argv)
{
	cv::Mat img_1 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img_2 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort2.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat K = ( cv::Mat_<double> (3, 3) << 458.654, 0.0,367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> (5, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 );

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);



	std::vector<cv::DMatch> matches;
	matcher->match(descriptors_1, descriptors_2, matches);


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

	std::vector <cv::Point2f> rancsac_result1, rancsac_result2;
	std::vector <cv::DMatch> rancsac_result_matches;
	int index = 0;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			rancsac_result1.push_back(p1[i]);
			rancsac_result2.push_back(p2[i]);
			good_matches[i].queryIdx = index;
			good_matches[i].trainIdx = index;
			rancsac_result_matches.push_back(good_matches[i]);
			index++;
		}
	}

    Eigen::Matrix<double, 8,9> A ;
    Eigen::Matrix<double, 9,1> e ;
    Eigen::Matrix<double, 3,3> E ;
    Eigen::Matrix<double, 3,3> sigma  = Eigen::MatrixXd::Zero(3,3);
    Eigen::Matrix<double, 3,3> t1_screw ;
    Eigen::Matrix<double, 3,3> t2_screw ;
    Eigen::Matrix<double, 3,3> R1 ;
    Eigen::Matrix<double, 3,3> R2 ;
    Eigen::Matrix<double, 9, 9> AtA = Eigen::MatrixXd::Zero(9,9);
    Eigen::Matrix3d R_z = Eigen::MatrixXd::Zero(3, 3);
    Eigen::Matrix3d R_z_neg = Eigen::MatrixXd::Zero(3, 3);

    std::set<int> random_number;
    int i = 0;

    R_z(0, 1) = 1;
    R_z(1, 0) = -1;
    R_z(2, 2) = 1;

    R_z_neg(0, 1) = -1;
    R_z_neg(1, 0) = 1;
    R_z_neg(2, 2) = 1;

    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    std::vector <cv::Point2f> p11, p21;
	
		p11.push_back(rancsac_result1[0]);
		p21.push_back(rancsac_result2[0]);
	
    for (int j=0; j< 8; j++){
        while(random_number.find(i) != random_number.end()){
            i = rand() % rancsac_result_matches.size();
        }
        p11.push_back(rancsac_result1[i]);
		p21.push_back(rancsac_result2[i]);
        random_number.insert(i);
        // std::cout << "i = " << i << "\n";
        // double x1 = rancsac_result1[i].x ;
        // double x2 = rancsac_result2[i].x ;
        // double y1 = rancsac_result1[i].y ;
        // double y2 = rancsac_result2[i].y ;
        double x1 = (rancsac_result1[i].x - cx) / fx;
        double x2 = (rancsac_result2[i].x - cx) / fx;
        double y1 = (rancsac_result1[i].y - cy) / fy;
        double y2 = (rancsac_result2[i].y - cy) / fy;
        A(j, 0) = x2 * x1 ;
        A(j, 1) = x2 * y1;
        A(j, 2) = x2 ;
        A(j, 3) = y2 * x1;
        A(j, 4) = y2 * y1;
        A(j, 5) = y2;
        A(j, 6) = x1 ;
        A(j, 7) = y1 ;
        A(j, 8) = 1;
    }
    // std::cout << A << std::endl;
    // auto U = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixU();
    auto V = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    // std::cout << V << std::endl;
    // // std::cout << A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).singularValues() << std::endl;
    e = V.col(8);

    E(0, 0) = e(0, 0);
    E(0, 1) = e(1, 0);
    E(0, 2) = e(2, 0);
    E(1, 0) = e(3, 0);
    E(1, 1) = e(4, 0);  
    E(1, 2) = e(5, 0);
    E(2, 0) = e(6, 0);
    E(2, 1) = e(7, 0);
    E(2, 2) = e(8, 0);


    std::cout <<"Essential Matrix \n"<< E;
    
    
    cv::Mat essentialMat = cv::findEssentialMat(p11, p21, K,cv::LMEDS);
    std::cout <<"\n Essential matrix from Opencv \n"<< essentialMat;
    Eigen::Matrix3d E_cv;
    cv::cv2eigen(essentialMat, E_cv);

    auto E_U = E_cv.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixU();
    auto E_V = E_cv.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixV();
    auto E_singular = E_cv.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).singularValues();

    sigma(0, 0) = E_singular(0);
    sigma(1, 1) = E_singular(1);
    sigma(2, 2) = E_singular(2);

    t1_screw = E_U * R_z *sigma * E_U.transpose();
    t2_screw = E_U * R_z_neg *sigma * E_U.transpose();
    R1 = E_U * R_z.transpose() * E_V.transpose();
    R2 = E_U * R_z_neg.transpose() * E_V.transpose();
    std::cout << "\nt1\n" << std::endl << t1_screw << std::endl;
    std::cout << "t2"<<std::endl<<t2_screw <<std::endl;
    std::cout << "R1 \n " << R1 << std::endl;
    std::cout << "R2 \n " << R2 << std::endl;

    // verify t and R using keypoint 7
    Eigen::Vector4d t1;
    Eigen::Vector4d t2;

    t1(0) = t1_screw(2, 1);
    t1(1) = t1_screw(0, 2);
    t1(2) = t1_screw(1, 0);
    t1(3) = 1;

    t2(0) = t2_screw(2, 1);
    t2(1) = t2_screw(0, 2);
    t2(2) = t2_screw(1, 0);
    t2(3) = 1;
    
    Eigen::Vector4d p_cam2;
    Eigen::Vector4d p_cam1;

    p_cam2(0) = (rancsac_result2[7].x - cx) / fx;
    p_cam2(1) = (rancsac_result2[7].y - cy) / fy;
    p_cam2(2) = 1;
    p_cam2(3) = 1;
    p_cam1(0) = (rancsac_result1[7].x - cx) / fx;
    p_cam1(1) = (rancsac_result1[7].y - cy) / fy;
    p_cam1(2) = 1;
    p_cam1(3) = 1;

    if (varify_R_t(p_cam1, p_cam2, R1, t1)){
        std::cout <<"The correct R and t are: " <<std::endl <<R1 << std::endl << std::endl << t1;
        return 0;
    }
    if (varify_R_t(p_cam1, p_cam2, R1, t2)){
        std::cout <<"The correct R and t are: " <<std::endl <<R1 << std::endl << std::endl << t2;
    }
    if (varify_R_t(p_cam1, p_cam2, R2, t1)){
        std::cout <<"The correct R and t are: " <<std::endl <<R2 << std::endl << std::endl << t1;
    }
    if (varify_R_t(p_cam1, p_cam2, R2, t2)){
        std::cout <<"The correct R and t are: " <<std::endl <<R2<< std::endl << std::endl << t2;
    }

    return 0;
}
