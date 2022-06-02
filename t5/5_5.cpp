
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

bool varify_R_t(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Matrix<double, 3,3> R,  Eigen::Vector3d t, cv::Mat K){
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R(0,0), R(0,1), R(0,2), t(0),  R(1,0), R(1,1), R(1,2), t(1), R(2,0), R(2,1), R(2,2), t(2));
    std::vector<cv::Point2d> pt1, pt2;
    cv::Mat pts_4d;
    pt1.push_back(cv::Point2d(p1(0), p1(1)));
    pt2.push_back(cv::Point2d(p2(0),p2(1)));
    cv::triangulatePoints(T1, T2,pt1, pt2, pts_4d);
    cv::Mat x = pts_4d.col(0);
    Eigen::MatrixXd A;
    cv::cv2eigen(x, A);
    A /= A(3);

    if (A(2) >0){
        if ((R * A.block<3, 1>(0, 0) + t)(2) > 0)
            return 1;
        
    }
    return 0;
}

int main(int argc, char **argv)
{
	cv::Mat img_1 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img_2 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_2.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat K = ( cv::Mat_<double> (3, 3) << 356.1094055175781, 0, 362.754261616093, 0.0,418.0326843261719, 250.1802333891737, 0.0, 0.0, 1.0 );
 
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
	cv::Mat Fundamental = cv::findFundamentalMat(p1, p2, RansacStatus, cv::FM_RANSAC,0.05,0.99);
   

    std::vector<cv::Point2f> rancsac_result1, rancsac_result2;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			rancsac_result1.push_back(p1[i]);
			rancsac_result2.push_back(p2[i]);
		}
	}
    std::cout << "num of corres" << rancsac_result1.size() << std::endl;

    Eigen::Matrix<double, 15,9> A ;
    Eigen::Matrix<double, 9,1> e ;
    Eigen::Matrix<double, 3,3> E ;
    Eigen::Matrix<double, 3,3> sigma  = Eigen::MatrixXd::Zero(3,3);
    Eigen::Matrix<double, 3,3> t1_screw ;
    Eigen::Matrix<double, 3,3> t2_screw ;
    Eigen::Matrix<double, 3,3> R1 ;
    Eigen::Matrix<double, 3,3> R2 ;
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
	
    for (int j=0; j< 15; j++){
        while(random_number.find(i) != random_number.end()){
            i = rand() % rancsac_result1.size();
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
    // auto V = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    // // std::cout << V << std::endl;
    Eigen::JacobiSVD<Eigen::Matrix<double, 15, 9>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 9> V = svd.matrixV();
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

    Eigen::Vector3d p_2;
    Eigen::Vector3d p_1;
    for (int i = 0; i < 15; i++){
    p_2(0) = (rancsac_result2[i].x - cx) / fx;
    p_2(1) = (rancsac_result2[i].y - cy) / fy;
    p_2(2) = 1;

    p_1(0) = (rancsac_result1[i].x - cx) / fx;
    p_1(1) = (rancsac_result1[i].y - cy) / fy;
    p_1(2) = 1;
    std::cout << "i = " << i << std::endl;
    std::cout << "Me " << p_2.transpose() * E * p_1 << std::endl;
    std::cout << "OpenCV " << p_2.transpose() * E_cv * p_1 << std::endl;
}
    auto E_U = E.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixU();
    auto E_V = E.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).matrixV();
    auto E_singular = E.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).singularValues();

    sigma(0, 0) = E_singular(0);
    sigma(1, 1) = E_singular(1);
    sigma(2, 2) = 0;

    t1_screw = E_U * R_z *sigma * E_U.transpose();
    t2_screw = E_U * R_z_neg *sigma * E_U.transpose();
    R1 = E_U * R_z.transpose() * E_V.transpose();
    R2 = E_U * R_z_neg.transpose() * E_V.transpose();

    Eigen::JacobiSVD<Eigen::Matrix<double, 3,3>> svd_R1(R1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::Matrix<double, 3,3>> svd_R2(R2, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d diag_R1 = Eigen::MatrixXd::Zero(3, 3);
    Eigen::Matrix3d diag_R2 = Eigen::MatrixXd::Zero(3, 3);
    diag_R1(0, 0) = 1;
    diag_R1(1, 1) = 1;
    diag_R1(2, 2) = R1.determinant();
    diag_R2(0, 0) = 1;
    diag_R2(1, 1) = 1;
    diag_R2(2, 2) = R2.determinant();

    // R1 = svd_R1.matrixU() *diag_R1*svd_R1.matrixV().transpose();
    // R2 = svd_R2.matrixU() *diag_R2*svd_R2.matrixV().transpose();

   



    // // verify t and R using keypoint 7
    Eigen::Vector3d t1;
    Eigen::Vector3d t2;

    t1(0) = t1_screw(2, 1);
    t1(1) = t1_screw(0, 2);
    t1(2) = t1_screw(1, 0);

    t2(0) = t2_screw(2, 1);
    t2(1) = t2_screw(0, 2);
    t2(2) = t2_screw(1, 0);
    std::cout << "\nt1\n" << std::endl << t1 << std::endl;
    std::cout << "t2"<<std::endl<<t2 <<std::endl;
    std::cout << "R1 \n " << R1 << std::endl;
    std::cout << "R2 \n " << R2 << std::endl;
    
    Eigen::Vector2d p_cam2;
    Eigen::Vector2d p_cam1;

    p_cam2(0) = (rancsac_result2[2].x - cx) / fx;
    p_cam2(1) = (rancsac_result2[2].y - cy) / fy;

    p_cam1(0) = (rancsac_result1[2].x - cx) / fx;
    p_cam1(1) = (rancsac_result1[2].y - cy) / fy;

    if (varify_R_t(p_cam1, p_cam2, R1, t1, K)){
        std::cout << "\nR = \n"
                  << R1 << "\n \n t = \n"
                  << t1;
    };
    if (varify_R_t(p_cam1, p_cam2, R1, t2, K)){
        std::cout << "\nR = \n"
                  << R1 << "\n \n t = \n"
                  << t2;
    };
    if (varify_R_t(p_cam1, p_cam2, R2, t1, K)){
        std::cout << "\nR = \n"
                  << R2 << "\n \n t = \n"
                  << t1;
    };
    if (varify_R_t(p_cam1, p_cam2, R2, t2, K)){
        std::cout << "\nR = \n"
                  << R2 << "\n \n t = \n"
                  << t2;
    };

    return 0;
}
