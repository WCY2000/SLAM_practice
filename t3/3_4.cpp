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
    Eigen::Matrix<double, 100, 2> A1 = Eigen::MatrixXd::Ones(100,2); 
    Eigen::Matrix<double, 100, 2> A2 = Eigen::MatrixXd::Ones(100,2); 
    Eigen::Matrix<double, 100, 1> b1;
    Eigen::Matrix<double, 100, 1> b2;
    double k1;
    double k2;


    std::ifstream infile;
    infile.open("/home/chenyu/Desktop/SLAM_practice/t3/data.txt");

    int i = 0;
    int j = 0;
    std::string tmp;
    infile >> tmp;
    infile >> tmp;

    while (infile >> tmp){
        if (j % 2 == 0){
            A1(i,0) = std::stod(tmp);
            j++;
        }
        else{
            b1(i) = std::stod(tmp);
            j++;
            i++;
        }
    }

    std::ifstream infile1;
    infile1.open("/home/chenyu/Desktop/SLAM_practice/t3/data2.txt");

    i = 0;
    j = 0;
    infile1 >> tmp;
    infile1 >> tmp;

    while (infile1 >> tmp){
        if (j % 2 == 0){
            A2(i,0) = std::stod(tmp);
            j++;
        }
        else{
            b2(i) = std::stod(tmp);
            j++;
            i++;
        }
    }


    // std::cout<<A;
    // std::cout<<b;
    // x = (ATA)-1ATb)
    x = (A2.transpose()*A2).inverse()*A2.transpose()*b2;
    std::cout<<"The solution of normal equation is: m = " << x(0) << ", n = " << x(1) << std::endl;

    x1 = A2.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(b2);
    std::cout<<"The solution of SVD decomposition is: m = " << x1(0) << ", n = "<< x1(1) << std::endl;
    
    x2 = A2.fullPivHouseholderQr().solve(b2);
    std::cout<<"The solution of QR decomposition is: m = " << x2(0) << ", n = "<< x2(1)<< std::endl;



    auto single_val1 = A1.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).singularValues();
    auto single_val2 = A2.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).singularValues();
    std::cout<<"Signle Values of Data 1: "<<single_val1<< std::endl;
    std::cout<<"Signle Values of Data 2: "<<single_val2<< std::endl;
    std::cout<<"Condition number of Data 1: "<<single_val1[0]/single_val1[1]<<std::endl;
    std::cout<<"Condition number of Data 2: "<<single_val2[0]/single_val2[1]<<std::endl;


    return 0;
}