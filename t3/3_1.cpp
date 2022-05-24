#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
using namespace std;
using namespace Eigen;


int main(){

    Eigen::Quaterniond imu_q_camL(Eigen::Vector4d(0.13431639597354814, 
                                                0.00095051670014565813,
                                                -0.0084222184858180373, 
                                                0.99090224973327068));

    Eigen::Vector3d imu_p_camL(-0.050720060477640147, 
                                -0.0017414170413474165, 
                                0.0022943667597148118);

    Eigen::Quaterniond imu_q_camR(Eigen::Vector4d(0.13492462817073628,
                                                -0.00013648999867379373,
                                                -0.015306242884176362,
                                                0.99073762672679389));
    Eigen::Vector3d imu_p_camL(0.051932496584961352, 
                            -0.0011555929083120534, 
                            0.0030949732069645722);



    return 0;
}