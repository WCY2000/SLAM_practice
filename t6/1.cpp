#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ceres/ceres.h"
#include "glog/logging.h"
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

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

Eigen::Matrix3d skew_matrix(Eigen::Vector3d p1){
    Eigen::Matrix3d p1_skew = Eigen::MatrixXd::Zero(3, 3);
    p1_skew(0, 1) = -p1(2);
    p1_skew(0, 2) = p1(1);
    p1_skew(1, 0) = p1(2);
    p1_skew(1, 2) = -p1(0);
    p1_skew(2, 0) = -p1(1);
    p1_skew(2, 1) = p1(0);
    return p1_skew;
}


Eigen::Vector3d triangulation(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Matrix3d R,  Eigen::Vector3d t){
    Eigen::Vector3d P;
    Eigen::Vector3d pt1;
    Eigen::Vector3d pt2;

    pt1(0) = p1(0);
    pt1(1) = p1(1);
    pt1(2) = 1;
    pt2(0) = p2(0);
    pt2(1) = p2(1);
    pt2(2) = 1;

    double s1, s2;
    auto m = -skew_matrix(pt2) * t;
    auto n = (skew_matrix(pt2) * R * pt1);
    if (n(0) == 0)
        s1 = 0;
    else {s1 = m(0) / n(0);}
    
    if (pt2(0) != 0)
        s2 = (s1 * R * pt1 + t)(0) / p2(0);
    else{
        if(pt2(1) != 0){
            s2 = (s1 * R * pt1 + t)(1) / pt2(1);
        }
    }
    std::cout << "\n\n Depth in Camera 1 is s1: " << s1 << " \n Depth in Camera 2 is s2: " << s2<<"\n";

    P<< p1(0)*s1, p1(1)*s1, s1;
    return P;
}


int main(){

    Eigen::Vector3d t ( 0.137705, -0.132793, -0.682211);
    Eigen::Matrix3d R;
    R << 0.955395, 0.0617142, -0.288812, -0.0388687, 0.995692, 0.084184,
                        0.292764, -0.0692032,   0.953677;

    cv::Mat img_1 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img_2 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_2.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat K = ( cv::Mat_<double> (3, 3) << 356.1094055175781, 0, 362.754261616093, 0.0,418.0326843261719, 250.1802333891737, 0.0, 0.0, 1.0 );
    
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

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
    std::cout<<"The number of Correspondences is: "<< rancsac_result1.size()<<std::endl;

    std::vector<Eigen::Vector3d> p3d;
    std::vector<Eigen::Vector2d> p2d;


    // for (int i=0; i< rancsac_result1.size(); i++){
    //     Eigen::Vector2d p_cam2;
    //     Eigen::Vector2d p_cam1;
    //     p_cam1(0) = (rancsac_result1[i].x - cx) / fx;
    //     p_cam1(1) = (rancsac_result1[i].y - cy) / fy;

    //     p_cam2(0) = (rancsac_result2[i].x - cx) / fx;
    //     p_cam2(1) = (rancsac_result2[i].y - cy) / fy;

    //     auto P_world = triangulation(p_cam1,p_cam2,R,t);
    //     p3d.push_back(P_world);

    // } 

    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R(0,0), R(0,1), R(0,2), t(0),  R(1,0), R(1,1), R(1,2), t(1), R(2,0), R(2,1), R(2,2), t(2));
    std::vector<cv::Point2d> pt1, pt2;
    cv::Mat pts_4d;

    


    for (int i=0; i< rancsac_result1.size(); i++){
        Eigen::Vector2d p_cam2;
        Eigen::Vector2d p_cam1;
        p_cam1(0) = (rancsac_result1[i].x - cx) / fx;
        p_cam1(1) = (rancsac_result1[i].y - cy) / fy;

        p_cam2(0) = (rancsac_result2[i].x - cx) / fx;
        p_cam2(1) = (rancsac_result2[i].y - cy) / fy;

        pt1.push_back(cv::Point2d(p_cam1(0), p_cam1(1)));
        pt2.push_back(cv::Point2d(p_cam2(0), p_cam2(1)));

    } 


    cv::triangulatePoints(T1, T2,pt1, pt2, pts_4d);

    Eigen::MatrixXd A;
    cv::cv2eigen(pts_4d, A);
    // std::cout<<"p2d: "<< std::endl;
    // for (auto i : p2d){
    //     std::cout<< i <<std::endl;
    // }

   
    ceres::Problem problem;
    Eigen::Quaterniond q(R);
    double c_rotation[2][4];
    double c_translation[2][3];
    double position_3d [rancsac_result1.size()][3];

    c_translation[1][0] = t(0);
	c_translation[1][1] = t(1);
    c_translation[1][2] = t(2);
	c_rotation[1][0] = q.w();
	c_rotation[1][1] = q.x();
	c_rotation[1][2] = q.y();
	c_rotation[1][3] = q.z();

    c_translation[0][0] = 0;
	c_translation[0][1] = 0;
    c_translation[0][2] = 0;
	c_rotation[0][0] = 1;
	c_rotation[0][1] = 0;
	c_rotation[0][2] = 0;
	c_rotation[0][3] = 0;
    
   
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    problem.AddParameterBlock(c_rotation[0], 4, local_parameterization);
	problem.AddParameterBlock(c_translation[0], 3);
    problem.AddParameterBlock(c_rotation[1], 4, local_parameterization);
	problem.AddParameterBlock(c_translation[1], 3);
    problem.SetParameterBlockConstant(c_rotation[1]);
    problem.SetParameterBlockConstant(c_translation[1]);


		for (int i = 0; i < 5; i++)
		{

			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												rancsac_result2[i].x,
												rancsac_result2[i].y);
            
            position_3d[i][0] = A.col(i)(0);
            position_3d[i][1] = A.col(i)(1);
            position_3d[i][2] = A.col(i)(2);

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[0], c_translation[0], 
    								position_3d[i]);	 
		}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
	options.max_solver_time_in_seconds = 10;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
    
    Eigen::Matrix3d R_optima ;
    Eigen::Quaterniond q_optima(c_rotation[0][0],c_rotation[0][1],c_rotation[0][2],c_rotation[0][3]);
    R_optima = q_optima.toRotationMatrix();
    std::cout<<"R \n"<<R_optima;


}