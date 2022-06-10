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
#include<cmath>

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

struct AnalyticCostFunction: public ceres::SizedCostFunction<2, 4, 3, 3>{
	AnalyticCostFunction(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

    virtual bool Evaluate(double const* const* parameters, double* residuals,double** jacobians) const{
		
        Eigen::Vector3d trans(parameters[1][0],parameters[1][1],parameters[1][2]);
        Eigen::Vector3d point_cam1(parameters[2][0],parameters[2][1],parameters[2][2]);
        Eigen::Vector3d point_cam2;
        Eigen::Quaterniond quat;
        quat.w() = parameters[0][0];
        quat.x() = parameters[0][1];
        quat.y() = parameters[0][2];
        quat.z() = parameters[0][3];
        point_cam2 = quat*point_cam1+ trans;
        
		double xp = point_cam2(0) / point_cam2(2);
    	double yp = point_cam2(1) / point_cam2(2);
    	residuals[0] = (xp - double(observed_u))*fx;
    	residuals[1] = (yp - double(observed_v))*fy;

        // std::cout<< "3d point x is: "<<point_cam2(0)<<" , y is: "<<point_cam2(1)<< " , z is: "<< point_cam2(2)<<std::endl;
        // std::cout<< "observation "<<observed_u <<"  "<<observed_v<<std::endl;

        double fx_by_z = fx / point_cam2(2);
        double fy_by_z = fy / point_cam2(2);

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
        double fx_by_zz = fx_by_z / point_cam2(2);
        double fy_by_zz = fy_by_z / point_cam2(2);

        J_cam << fx_by_z, 0, - fx_by_zz * point_cam2(0),
                0, fy_by_z, - fy_by_zz * point_cam2(1);
    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();

            J_se3.block<2,3>(0,1) =  -J_cam  * skew_matrix( point_cam1 ) ;
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = J_cam ;
        }
        if(jacobians[2] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[2]);
            J_point = J_cam * quat.toRotationMatrix();
        }
    }

    return true;
}
	double observed_u;
	double observed_v;
    double fx = 356.1094055175781;
    double fy = 418.0326843261719;
    double cx = 362.754261616093;
    double cy = 250.1802333891737;
};

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

	ceres::LocalParameterization* local_parameterization1 = new ceres::QuaternionParameterization();
    ceres::LocalParameterization* local_parameterization2 = new ceres::QuaternionParameterization();
    problem.AddParameterBlock(c_rotation[0], 4, local_parameterization1);
	problem.AddParameterBlock(c_translation[0], 3);
    problem.AddParameterBlock(c_rotation[1], 4, local_parameterization2);
	problem.AddParameterBlock(c_translation[1], 3);
    problem.SetParameterBlockConstant(c_rotation[0]);
    problem.SetParameterBlockConstant(c_translation[0]);    

		for (int i = 0; i < rancsac_result1.size(); i++)
		{
			auto cost_function = new AnalyticCostFunction(
												(rancsac_result1[i].x - cx) / fx,
												(rancsac_result1[i].y - cy) / fy);
            position_3d[i][0] = A.col(i)(0);
            position_3d[i][1] = A.col(i)(1);
            position_3d[i][2] = A.col(i)(2);

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[0],c_translation[0], 
    								position_3d[i]);
        }


        for (int i = 0; i < rancsac_result1.size(); i++)
		{

			auto cost_function = new AnalyticCostFunction(
												(rancsac_result2[i].x - cx) / fx,
												(rancsac_result2[i].y - cy) / fy);
            position_3d[i][0] = A.col(i)(0);
            position_3d[i][1] = A.col(i)(1);
            position_3d[i][2] = A.col(i)(2);

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[1],c_translation[1], 
    								position_3d[i]);
		}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
    options.check_gradients = true;
    options.gradient_check_relative_precision = 2;

	options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 3000;
	options.max_solver_time_in_seconds = 10;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
    
    
    Eigen::Matrix3d R_optima ;
    Eigen::Vector3d t_optima(c_translation[1][0],c_translation[1][1],c_translation[1][2]) ;
    Eigen::Quaterniond q_optima(c_rotation[1][0],c_rotation[1][1],c_rotation[1][2],c_rotation[1][3]);
    // Eigen::Vector3d t_optima(c_translation[0][0],c_translation[0][1],c_translation[0][2]) ;
    // Eigen::Quaterniond q_optima(c_rotation[0][0],c_rotation[0][1],c_rotation[0][2],c_rotation[0][3]);
    R_optima = q_optima.toRotationMatrix();
    std::vector<Eigen::Vector3d> point_optima;

    std::cout<<"\n R after Ceres optimization \n"<<R_optima;
    std::cout<<"\n t after Ceres optimization \n"<<t_optima;

    for (int i=0; i< rancsac_result2.size(); i++){
        point_optima.emplace_back(Eigen::Vector3d(position_3d[i][0],position_3d[i][1],position_3d[i][2]));
    }

    double residual1[rancsac_result1.size()];
    double residual2[rancsac_result1.size()];
    double sum1 = 0;
    double mean1 = 0;
    double mse1 = 0;
    double rmse1 = 0;
    double sum2 = 0;
    double mean2 = 0;
    double mse2 = 0;
    double rmse2 = 0;
    std::vector<cv::Point2d> reprojectedPoints1;
    std::vector<cv::Point2d> reprojectedPoints2;
    for (int i = 0; i< rancsac_result1.size(); i++){

        Eigen::Vector3d p_new = R_optima * point_optima[i]+ t_optima;
        double residual_x1 = (rancsac_result1[i].x - cx) / fx - point_optima[i](0)/point_optima[i](2);
        double residual_y1 = (rancsac_result1[i].y - cy) / fy - point_optima[i](1)/point_optima[i](2);
        // std::cout<<"residual 1: "<< (rancsac_result1[i].x - cx) / fx <<" - "<< point_optima[i](0)/point_optima[i](2)<<std::endl;
        double residual_x2 = (rancsac_result2[i].x - cx) / fx - p_new(0)/p_new(2);
        double residual_y2 = (rancsac_result2[i].y - cy) / fy - p_new(1)/p_new(2);
        residual1[i] = sqrt(residual_x1*residual_x1 + residual_y1*residual_y1);
        residual2[i] = sqrt(residual_x2*residual_x2 + residual_y2*residual_y2);
        sum1 += residual1[i];
        sum2 += residual2[i];

        reprojectedPoints1.push_back(cv::Point2d (point_optima[i](0)/point_optima[i](2)*fx+cx, point_optima[i](1)/point_optima[i](2)*fy+cy));
        reprojectedPoints2.push_back(cv::Point2d (p_new(0)/p_new(2)*fx+cx, p_new(1)/p_new(2)*fy+cy));

    }
    mean1 = sum1 / rancsac_result1.size();
    mean2 = sum2 / rancsac_result2.size();



    for (int i = 0; i<rancsac_result1.size(); i++){
        mse1 += (residual1[i] - mean1)* (residual1[i] - mean1);
        mse2 += (residual2[i] - mean2)* (residual2[i] - mean2);
    }

    mse1 /= rancsac_result1.size();
    rmse1 = sqrt(mse1);
    mse2 /= rancsac_result2.size();
    rmse2 = sqrt(mse2);
    std::cout<<"\nThe mean of residual in image 1 is: "<< mean1 <<std::endl;
    std::cout<<"The mean square error of residual in image 1 is: "<< mse1 <<std::endl;
    std::cout<<"The root of  square error of residual in image 1 is: "<< rmse1 <<std::endl;

    std::cout<<"\nThe mean of residual in image 2 is: "<< mean2 <<std::endl;
    std::cout<<"The mean square error of residual in image 2 is: "<< mse2 <<std::endl;
    std::cout<<"The root of  square error of residual in image 2 is: "<< rmse2 <<std::endl;


    cv::Mat outimg1 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_1.png");
	cv::Mat outimg2 = cv::imread("/home/chenyu/Desktop/SLAM_practice/t5/undistort_2.png");
    
    // cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    for (int i=0; i< rancsac_result1.size(); i++){
        cv::circle(outimg1, rancsac_result1[i], 1, cv::Scalar(255, 0, 120), -1);
        cv::circle(outimg1, reprojectedPoints1[i], 1, cv::Scalar(0, 255, 120), -1);
    }
    cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t6/reprojection1_analysis.jpg", outimg1);
	// cv::drawKeypoints(img_2, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        for (int i=0; i< rancsac_result1.size(); i++){
        cv::circle(outimg2, rancsac_result2[i], 1, cv::Scalar(255, 0, 120), -1);
        cv::circle(outimg2, reprojectedPoints2[i], 1, cv::Scalar(0, 255, 120), -1);
    }
	cv::imwrite("/home/chenyu/Desktop/SLAM_practice/t6/reprojection2_analysis.jpg", outimg2);




}