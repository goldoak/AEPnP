#ifndef ANISOTROPICEPNP_HPP_
#define ANISOTROPICEPNP_HPP_

#include <stdlib.h>
#include <Eigen/Eigen>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>


class AnisotropicEpnp
{
public:
  AnisotropicEpnp(void);
  ~AnisotropicEpnp();

  void set_maximum_number_of_correspondences(const int n);
  void reset_correspondences(void);
  void add_correspondence(
      const double X,
      const double Y,
      const double Z,
      const double x,
      const double y,
      const double z);

  double compute_pose(double R[3][3], double s[3], double T[3]);
  void compute_pose( Eigen::Matrix3d & R, Eigen::Vector3d & s, Eigen::Vector3d & t );

private:
  void choose_control_points(void);
  void compute_barycentric_coordinates(void);
  void fill_M(
      Eigen::MatrixXd & M,
      const int row,
      const double * alphas,
      const double u,
      const double v);

  double uc, vc, fu, fv;
  double * pws, * us, * alphas, * pcs;
  int * signs; //added!
  int maximum_number_of_correspondences;
  int number_of_correspondences;

  double cws[4][3], ccs[4][3];
  double cws_determinant;
};

#endif /* ANISOTROPICEPNP_HPP_ */
