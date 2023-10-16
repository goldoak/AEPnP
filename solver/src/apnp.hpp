#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include "anisotropicPnP.hpp"

inline void fill3x12( const Eigen::Vector3d & x, Eigen::Matrix<double,3,12> & Phi )
{
    double x1 = x(0,0);
    double x2 = x(1,0);
    double x3 = x(2,0);

    Phi = Eigen::Matrix<double,3,12>::Zero();

    Phi(0,0)  =  x1;
    Phi(0,3)  =  x2;
    Phi(0,6)  =  x3;
    Phi(0,7)  = -x3;

    Phi(1,1)  =  x1;
    Phi(1,4)  =  x2;
    Phi(1,8)  =  x3;
    Phi(1,9)  = -x3;

    Phi(2,2)  =  x1;
    Phi(2,5)  =  x2;
    Phi(2,10) =  x3;
    Phi(2,11) = -x3;
}


inline void fill3x12_focal( const Eigen::Vector3d & x, Eigen::Matrix<double,3,12> & Phi )
{
    double x1 = x(0,0);
    double x2 = x(1,0);
    double x3 = x(2,0);

    Phi = Eigen::Matrix<double,3,12>::Zero();

    Phi(0,0)  =  x1;
    Phi(0,1)  =  x2;
    Phi(0,2)  =  x3;

    Phi(1,3)  =  x1;
    Phi(1,4)  =  x2;
    Phi(1,5)  =  x3;

    Phi(2,6)  =  x1;
    Phi(2,7)  =  -x1;
    Phi(2,8) =  x2;
    Phi(2,9) = -x2;
    Phi(2,10) = x3;
    Phi(2,11) = -x3;
}


inline auto BaseSolver(Eigen::Matrix<double, Eigen::Dynamic, 3> & bv, Eigen::Matrix<double, Eigen::Dynamic, 3> & wp,
                       Eigen::Matrix3d & H, Eigen::MatrixXd & M, int case_id=0)
{

    //the needed input is 3D points (vector<Vector3d> wp)
    //and bearing vectors (unit norm direction vectors seen from the camera frame (vector<Vector3d> bv)
    // inv(K) * [u;v;1] / norm(inv(K) * [u;v;1])

    int numberBearingVectors = bv.rows();

    //derive M
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    for( int i = 0; i < numberBearingVectors; i++ ) {
        F += bv.row(i).transpose() * bv.row(i);
    }

    Eigen::Matrix3d H_inv = (double) numberBearingVectors * Eigen::Matrix3d::Identity() - F;
    H = H_inv.inverse();

    Eigen::Matrix<double,3,12> I = Eigen::Matrix<double,3,12>::Zero();
    Eigen::Matrix<double,3,12> Phi;

    for( int i = 0; i < numberBearingVectors; i++ ) {
        Eigen::Matrix3d Vk = H * ( bv.row(i).transpose() * bv.row(i) - Eigen::Matrix3d::Identity() );
//        fill3x12(wp.row(i),Phi);
        if (case_id == 0) {fill3x12(wp.row(i),Phi);}
        else if (case_id == 1) {fill3x12_focal(wp.row(i),Phi);}
        else {std::cout << "Wrong case number!";}
        I += Vk * Phi;
    }

    M = Eigen::MatrixXd::Zero(12,12);
    for( int i = 0; i < numberBearingVectors; i++ )
    {
//        fill3x12(wp.row(i),Phi);
        if (case_id == 0) {fill3x12(wp.row(i),Phi);}
        else if (case_id == 1) {fill3x12_focal(wp.row(i),Phi);}
        else {std::cout << "Wrong case number!";}
        Eigen::Matrix3d temp = bv.row(i).transpose() * bv.row(i) - Eigen::Matrix3d::Identity();
        Eigen::Matrix<double,3,12> Ai =  temp * ( Phi + I);
        M += (Ai.transpose() * Ai);
    }
}


inline auto APnPSolver(Eigen::Matrix<double, Eigen::Dynamic, 3> & bv, Eigen::Matrix<double, Eigen::Dynamic, 3> & wp)
{
    Eigen::Matrix3d H;
    Eigen::MatrixXd M;
    BaseSolver(bv, wp, H, M, 0);

    std::vector< Eigen::Matrix<double,7,1>, Eigen::aligned_allocator<Eigen::Matrix<double,7,1> > > solutions;
    polyjam::anisotropicPnP::solve( M, solutions );

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > T;
    for( int i = 0; i < solutions.size(); i++ )
    {
        Eigen::Matrix<double,7,1> solution;
        solution << solutions[i][4], solutions[i][5], solutions[i][6], solutions[i][0], solutions[i][1], solutions[i][2], solutions[i][3];

        double u = solution(0,0);
        double v = solution(1,0);
        double w = solution(2,0);
        double a = solution(3,0);
        double b = solution(4,0);
        double c = solution(5,0);
        double s = solution(6,0);

        Eigen::Matrix3d scaled_Rotation;
        scaled_Rotation << u, s*a, v*c-w*b,
                           v, s*b, w*a-u*c,
                           w, s*c, u*b-v*a;

        int numberBearingVectors = bv.rows();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        for( int i = 0; i < (int) numberBearingVectors; i++ ) {
            t += H * ( bv.row(i).transpose() * bv.row(i) - Eigen::Matrix3d::Identity() ) * scaled_Rotation * wp.row(i).transpose();
        }

        Eigen::Transform<double, 3, Eigen::Affine> transform;
        transform.setIdentity();
        transform.translate(t);
        transform.rotate(scaled_Rotation);

        T.push_back(transform.matrix());
    }
    return T;
}


inline auto APnPSolver2(Eigen::Matrix<double, Eigen::Dynamic, 3> & bv, Eigen::Matrix<double, Eigen::Dynamic, 3> & wp)
{
    Eigen::Matrix3d H;
    Eigen::MatrixXd M;
    BaseSolver(bv, wp, H, M, 1);

    std::vector< Eigen::Matrix<double,7,1>, Eigen::aligned_allocator<Eigen::Matrix<double,7,1> > > solutions;
    polyjam::anisotropicPnP::solve( M, solutions );

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > T;
    for( int i = 0; i < solutions.size(); i++ )
    {
        Eigen::Matrix<double,7,1> solution;
        solution << solutions[i][4], solutions[i][5], solutions[i][6], solutions[i][0], solutions[i][1], solutions[i][2], solutions[i][3];

        double u = solution(0,0);
        double v = solution(1,0);
        double w = solution(2,0);
        double a = solution(3,0);
        double b = solution(4,0);
        double c = solution(5,0);
        double s = solution(6,0);

        Eigen::Matrix3d scaled_Rotation;
        scaled_Rotation << u, v, w,
                           s*a, s*b, s*c,
                           v*c-w*b, w*a-u*c, u*b-v*a;

        int numberBearingVectors = bv.rows();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        for( int i = 0; i < (int) numberBearingVectors; i++ ) {
            t += H * ( bv.row(i).transpose() * bv.row(i) - Eigen::Matrix3d::Identity() ) * scaled_Rotation * wp.row(i).transpose();
        }

        Eigen::Transform<double, 3, Eigen::Affine> transform;
        transform.setIdentity();
        transform.translate(t);
        transform.rotate(scaled_Rotation);

        T.push_back(transform.matrix());
    }
    return T;
}
