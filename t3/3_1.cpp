#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>

int main(){
//convert from JPL to Hamilton: x, y, z, w -> w, x, y,-z
    Eigen::Quaterniond imu_q_camL(Eigen::Vector4d(0.13431639597354814, 
                                                0.00095051670014565813,
                                                0.0084222184858180373, 
                                                0.99090224973327068));

    Eigen::Vector3d imu_p_camL(-0.050720060477640147, 
                                -0.0017414170413474165, 
                                0.0022943667597148118);

    Eigen::Quaterniond imu_q_camR(Eigen::Vector4d(0.13492462817073628,
                                                -0.00013648999867379373,
                                                0.015306242884176362,
                                                0.99073762672679389));
    Eigen::Vector3d imu_p_camR(0.051932496584961352, 
                            -0.0011555929083120534, 
                            0.0030949732069645722);

    Eigen::Matrix4d imu_T_camL;
    Eigen::Matrix4d imu_T_camR;

    imu_q_camL.normalize();
    imu_q_camR.normalize();

    imu_T_camL.block(0,0,3,3) = imu_q_camL.matrix();
    imu_T_camL.block(0,3,3,1) = imu_p_camL;
    imu_T_camL(3,0) = 0;
    imu_T_camL(3,1) = 0;
    imu_T_camL(3,2) = 0;
    imu_T_camL(3,3) = 1;
    // std::cout<<imu_T_camL
    imu_T_camR.block(0,0,3,3) = imu_q_camR.matrix();
    imu_T_camR.block(0,3,3,1) = imu_p_camR;
    imu_T_camR(3,0) = 0;
    imu_T_camR(3,1) = 0;
    imu_T_camR(3,2) = 0;
    imu_T_camR(3,3) = 1;
    // std::cout<<imu_T_camR

    Eigen::Matrix4d camL_T_camR = imu_T_camL.inverse() * imu_T_camR;
    std::cout<<"transformation matrix from Camera Right to Camera Left is:" << std::endl << camL_T_camR;
    return 0;
}