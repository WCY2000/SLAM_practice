#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <string>

int main(){
// y = mx +n -> AX = B 

    Eigen::Vector2d x;
    Eigen::Vector2d x1;
    Eigen::Vector2d x2;
    Eigen::Matrix<double, 100, 2> A = Eigen::MatrixXd::Ones(100,2); 
    Eigen::Matrix<double, 100, 1> b;
    std::ifstream infile;
    infile.open("/home/chenyu/Desktop/SLAM_practice/t3/data.txt");

    int i = 0;
    int j = 0;
    std::string tmp;
    infile >> tmp;
    infile >> tmp;

    while (infile >> tmp){
        if (j % 2 == 0){
            A(i,0) = std::stod(tmp);
            j++;
        }
        else{
            b(i) = std::stod(tmp);
            j++;
            i++;
        }
    }
    // std::cout<<A;
    // std::cout<<b;
    // // x = (ATA)-1ATb)
    x = (A.transpose()*A).inverse()*A.transpose()*b;
    std::cout<<"The solution of normal equation is: m = " << x(0) << ", n = " << x(1) << std::endl;

    x1 = A.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(b);
    std::cout<<"The solution of SVD decomposition is: m = " << x1(0) << ", n = "<< x1(1) << std::endl;

    x2 = A.fullPivHouseholderQr().solve(b);
    std::cout<<"The solution of QR decomposition is: m = " << x2(0) << ", n = "<< x2(1);
    return 0;
}