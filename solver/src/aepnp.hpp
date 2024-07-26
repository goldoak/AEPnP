#include <cassert>
#include <Eigen/Eigen>
#include "AnisotropicEpnp.hpp"


inline Eigen::Matrix4d AEPnPSolver(Eigen::Matrix<double, Eigen::Dynamic, 2> & p2d, Eigen::Matrix<double, Eigen::Dynamic, 3> & p3d) {
    assert(p2d.rows() == p3d.rows() && "2D-3D correspondences must match!");

    int numberPairs = p2d.rows();
    
    AnisotropicEpnp aepnp;
    aepnp.set_maximum_number_of_correspondences(numberPairs);
    for( int i = 0; i < numberPairs; i++ ) {
        aepnp.add_correspondence(p3d(i, 0), p3d(i, 1), p3d(i, 2), p2d(i, 0), p2d(i, 1), 1.0);
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector3d scale;
    aepnp.compute_pose(R, scale, t);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    T.col(0) *= scale(0, 0);
    T.col(1) *= scale(1, 0);
    T.col(2) *= scale(2, 0);

    return T;
}