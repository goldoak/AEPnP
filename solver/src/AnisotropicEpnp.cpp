#include <iostream>
using namespace std;

#include "AnisotropicEpnp.hpp"


AnisotropicEpnp::AnisotropicEpnp(void)
{
  maximum_number_of_correspondences = 0;
  number_of_correspondences = 0;

  pws = 0;
  us = 0;
  alphas = 0;
  pcs = 0;
  signs = 0; //added

  this->uc = 0.0;
  this->vc = 0.0;
  this->fu = 1.0;
  this->fv = 1.0;
}

AnisotropicEpnp::~AnisotropicEpnp()
{
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
  delete [] signs; //added
}

void
AnisotropicEpnp::
    set_maximum_number_of_correspondences(int n)
{
  if (maximum_number_of_correspondences < n)
  {
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;
    if (signs != 0) delete [] signs; //added

    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
    signs = new int[maximum_number_of_correspondences];
  }
}

void
AnisotropicEpnp::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

void
AnisotropicEpnp::add_correspondence(
    double X,
    double Y,
    double Z,
    double x,
    double y,
    double z) //changed this interface
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = x/z;
  us[2 * number_of_correspondences + 1] = y/z;

  //added the following
  if(z > 0.0)
    signs[number_of_correspondences] = 1;
  else
    signs[number_of_correspondences] = -1;

  number_of_correspondences++;
}

void
AnisotropicEpnp::choose_control_points(void)
{
  cws[0][0] = 0; cws[0][1] = 0; cws[0][2] = 0;
  cws[1][0] = 1; cws[1][1] = 0; cws[1][2] = 0;
  cws[2][0] = 0; cws[2][1] = 1; cws[2][2] = 0;
  cws[3][0] = 0; cws[3][1] = 0; cws[3][2] = 1;
}

void
AnisotropicEpnp::
    compute_barycentric_coordinates(void)
{
  Eigen::Matrix3d CC;

  for(int i = 0; i < 3; i++)
    for(int j = 1; j < 4; j++)
      CC(i,j-1) = cws[j][i] - cws[0][i];

  Eigen::Matrix3d CC_inv = CC.inverse();

  for(int i = 0; i < number_of_correspondences; i++)
  {
    double * pi = pws + 3 * i;
    double * a = alphas + 4 * i;

    for(int j = 0; j < 3; j++)
      a[1 + j] =
        CC_inv(j,0) * (pi[0] - cws[0][0]) +
        CC_inv(j,1) * (pi[1] - cws[0][1]) +
        CC_inv(j,2) * (pi[2] - cws[0][2]);
    a[0] = 1.0f - a[1] - a[2] - a[3];
  }
}

void
AnisotropicEpnp::fill_M(
    Eigen::MatrixXd & M,
    const int row,
    const double * as,
    const double u,
    const double v)
{
  for(int i = 0; i < 4; i++)
  {
    M(row,3*i) = as[i] * fu;
    M(row,3*i+1) = 0.0;
    M(row,3*i+2) = as[i] * (uc - u);

    M(row+1,3*i) = 0.0;
    M(row+1,3*i+1) = as[i] * fv;
    M(row+1,3*i+2) = as[i] * (vc - v);
  }
}

double
AnisotropicEpnp::compute_pose(
    double R[3][3],
    double s[3],
    double t[3])
{
  choose_control_points();
  compute_barycentric_coordinates();

  Eigen::MatrixXd M(2*number_of_correspondences,12);

  for(int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

  Eigen::MatrixXd MtM = M.transpose() * M;
  Eigen::JacobiSVD< Eigen::MatrixXd > SVD(
      MtM,
      Eigen::ComputeFullV | Eigen::ComputeFullU );
  Eigen::MatrixXd Ut = SVD.matrixU().transpose();

  t[0]    = Ut(11,3 * 0     );
  t[1]    = Ut(11,3 * 0 + 1 );
  t[2]    = Ut(11,3 * 0 + 2 );
  R[0][0] = Ut(11,3 * 1     ) - Ut(11,3 * 0     );
  R[1][0] = Ut(11,3 * 1 + 1 ) - Ut(11,3 * 0 + 1 );
  R[2][0] = Ut(11,3 * 1 + 2 ) - Ut(11,3 * 0 + 2 );
  R[0][1] = Ut(11,3 * 2     ) - Ut(11,3 * 0     );
  R[1][1] = Ut(11,3 * 2 + 1 ) - Ut(11,3 * 0 + 1 );
  R[2][1] = Ut(11,3 * 2 + 2 ) - Ut(11,3 * 0 + 2 ); 
  R[0][2] = Ut(11,3 * 3     ) - Ut(11,3 * 0     );
  R[1][2] = Ut(11,3 * 3 + 1 ) - Ut(11,3 * 0 + 1 );
  R[2][2] = Ut(11,3 * 3 + 2 ) - Ut(11,3 * 0 + 2 );
  s[0]    = sqrt(R[0][0]*R[0][0] + R[1][0]*R[1][0] + R[2][0]*R[2][0]);
  s[1]    = sqrt(R[0][1]*R[0][1] + R[1][1]*R[1][1] + R[2][1]*R[2][1]);
  s[2]    = sqrt(R[0][2]*R[0][2] + R[1][2]*R[1][2] + R[2][2]*R[2][2]);
  R[0][0] /= s[0];
  R[1][0] /= s[0];
  R[2][0] /= s[0];
  R[0][1] /= s[1];
  R[1][1] /= s[1];
  R[2][1] /= s[1];
  R[0][2] /= s[2];
  R[1][2] /= s[2];
  R[2][2] /= s[2];

  return 0.0;
}

void
AnisotropicEpnp::compute_pose( Eigen::Matrix3d & R, Eigen::Vector3d & s, Eigen::Vector3d & t ) {

  double Rplain[3][3];
  double splain[3];
  double tplain[3];

  compute_pose( Rplain, splain, tplain );

  t(0) = tplain[0];
  t(1) = tplain[1];
  t(2) = tplain[2];
  s(0) = splain[0];
  s(1) = splain[1];
  s(2) = splain[2];
  R(0,0) = Rplain[0][0];
  R(1,0) = Rplain[1][0];
  R(2,0) = Rplain[2][0];
  R(0,1) = Rplain[0][1];
  R(1,1) = Rplain[1][1];
  R(2,1) = Rplain[2][1];
  R(0,2) = Rplain[0][2];
  R(1,2) = Rplain[1][2];
  R(2,2) = Rplain[2][2];

  //take care of solution having possibly wrong sign
  if(R.determinant() < 0) {
    R *= -1.0;
    t *= -1.0;
  }

  //somehow normalize the scale
  t /= s(0);
  s(1) /= s(0);
  s(2) /= s(0);
  s(0) = 1.0;
}
