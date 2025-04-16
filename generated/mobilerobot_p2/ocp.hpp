// This file was automatically generated by autogenu-jupyter (https://github.com/mayataka/autogenu-jupyter). 
// The autogenu-jupyter copyright holders make no ownership claim of its contents. 

#ifndef CGMRES__OCP_MOBILEROBOT_P2_HPP_ 
#define CGMRES__OCP_MOBILEROBOT_P2_HPP_ 
 
#define _USE_MATH_DEFINES

#include <cmath>
#include <array>
#include <iostream>

#include "cgmres/types.hpp"
#include "cgmres/detail/macros.hpp"

namespace cgmres {

/// 
/// @class OCP_mobilerobot_p2
/// @brief Definition of the optimal control problem (OCP) of mobilerobot_p2.
/// 
class OCP_mobilerobot_p2 { 
public:
  ///
  /// @brief Dimension of the state. 
  ///
  static constexpr int nx = 12;
 
  ///
  /// @brief Dimension of the control input. 
  ///
  static constexpr int nu = 4;
 
  ///
  /// @brief Dimension of the equality constraints. 
  ///
  static constexpr int nc = 2;
 
  ///
  /// @brief Dimension of the Fischer-Burmeister function (already counded in nc). 
  ///
  static constexpr int nh = 2;
 
  ///
  /// @brief Dimension of the concatenation of the control input and equality constraints. 
  ///
  static constexpr int nuc = nu + nc;

  ///
  /// @brief Dimension of the bound constraints on the control input. 
  ///
  static constexpr int nub = 4;

  double vx_ref = 0.4;
  double D = 0.5;
  double v_min = -0.5;
  double v_max = 0.5;
  double w_min = -0.75;
  double w_max = 0.75;
  double X_1 = 1;
  double Y_1 = 0.25;
  double R_1 = 0.5;
  double X_2 = 2;
  double Y_2 = -0.25;
  double R_2 = 0.5;

  std::array<double, 5> q = {10, 1, 0.01, 10, 0.01};
  std::array<double, 4> r = {0.1, 0.1, 1, 1};

  static constexpr std::array<int, nub> ubound_indices = {0, 1, 2, 3};
  std::array<double, nub> umin = {-1.0, -1.0, -1.0, -1.0};
  std::array<double, nub> umax = {1.0, 1.0, 1.0, 1.0};
  std::array<double, nub> dummy_weight = {0.1, 0.1, 0.1, 0.1};

  std::array<double, nh> fb_eps = {0.05, 0.05};

  void disp(std::ostream& os) const {
    os << "OCP_mobilerobot_p2:" << std::endl;
    os << "  nx:  " << nx << std::endl;
    os << "  nu:  " << nu << std::endl;
    os << "  nc:  " << nc << std::endl;
    os << "  nh:  " << nh << std::endl;
    os << "  nuc: " << nuc << std::endl;
    os << "  nub: " << nub << std::endl;
    os << std::endl;
    os << "  vx_ref: " << vx_ref << std::endl;
    os << "  D: " << D << std::endl;
    os << "  v_min: " << v_min << std::endl;
    os << "  v_max: " << v_max << std::endl;
    os << "  w_min: " << w_min << std::endl;
    os << "  w_max: " << w_max << std::endl;
    os << "  X_1: " << X_1 << std::endl;
    os << "  Y_1: " << Y_1 << std::endl;
    os << "  R_1: " << R_1 << std::endl;
    os << "  X_2: " << X_2 << std::endl;
    os << "  Y_2: " << Y_2 << std::endl;
    os << "  R_2: " << R_2 << std::endl;
    os << std::endl;
    Eigen::IOFormat fmt(4, 0, ", ", "", "[", "]");
    Eigen::IOFormat intfmt(1, 0, ", ", "", "[", "]");
    os << "  q: " << Map<const VectorX>(q.data(), q.size()).transpose().format(fmt) << std::endl;
    os << "  r: " << Map<const VectorX>(r.data(), r.size()).transpose().format(fmt) << std::endl;
    os << std::endl;
    os << "  ubound_indices: " << Map<const VectorXi>(ubound_indices.data(), ubound_indices.size()).transpose().format(intfmt) << std::endl;
    os << "  umin: " << Map<const VectorX>(umin.data(), umin.size()).transpose().format(fmt) << std::endl;
    os << "  umax: " << Map<const VectorX>(umax.data(), umax.size()).transpose().format(fmt) << std::endl;
    os << "  dummy_weight: " << Map<const VectorX>(dummy_weight.data(), dummy_weight.size()).transpose().format(fmt) << std::endl;
    os << std::endl;
    os << "  fb_eps: " << Map<const VectorX>(fb_eps.data(), fb_eps.size()).transpose().format(fmt) << std::endl;
  }

  friend std::ostream& operator<<(std::ostream& os, const OCP_mobilerobot_p2& ocp) { 
    ocp.disp(os);
    return os;
  }


  ///
  /// @brief Synchrozies the internal parameters of this OCP with the external references.
  /// This method is called at the beginning of each MPC update.
  ///
  void synchronize() {
  }

  ///
  /// @brief Computes the state equation dx = f(t, x, u).
  /// @param[in] t Time.
  /// @param[in] x State.
  /// @param[in] u Control input.
  /// @param[out] dx Evaluated value of the state equation.
  /// @remark This method is intended to be used inside of the cgmres solvers and does not check size of each argument. 
  /// Use the overloaded method if you call this outside of the cgmres solvers. 
  ///
  void eval_f(const double t, const double* x, const double* u, 
              double* dx) const {
    const double x0 = u[0]*cos(x[2]);
    const double x1 = u[0]*sin(x[2]);
    const double x2 = u[2]*cos(x[5]);
    const double x3 = u[2]*sin(x[5]);
    dx[0] = x0;
    dx[1] = x1;
    dx[2] = u[1];
    dx[3] = x2;
    dx[4] = x3;
    dx[5] = u[3];
    dx[6] = x0;
    dx[7] = x1;
    dx[8] = u[1];
    dx[9] = x2;
    dx[10] = x3;
    dx[11] = u[3];
 
  }

  ///
  /// @brief Computes the partial derivative of terminal cost with respect to state, 
  /// i.e., phix = dphi/dx(t, x).
  /// @param[in] t Time.
  /// @param[in] x State.
  /// @param[out] phix Evaluated value of the partial derivative of terminal cost.
  /// @remark This method is intended to be used inside of the cgmres solvers and does not check size of each argument. 
  /// Use the overloaded method if you call this outside of the cgmres solvers. 
  ///
  void eval_phix(const double t, const double* x, double* phix) const {
    const double x0 = x[0] - x[3];
    const double x1 = pow(x0, 2);
    const double x2 = (1.0/2.0)*q[3];
    const double x3 = sqrt(x1)*x2;
    const double x4 = 2*x[1] - 2*x[4];
    phix[0] = (1.0/2.0)*q[0]*(-2*t*vx_ref + 2*x[0]);
    phix[1] = q[1]*x[1];
    phix[2] = q[2]*x[2];
    phix[3] = 0;
    phix[4] = 0;
    phix[5] = 0;
    phix[6] = x3/x0;
    phix[7] = x2*x4;
    phix[8] = 0;
    phix[9] = -x0*x3/x1;
    phix[10] = -x2*x4;
    phix[11] = 0;
 
  }

  ///
  /// @brief Computes the partial derivative of the Hamiltonian with respect to state, 
  /// i.e., hx = dH/dx(t, x, u, lmd).
  /// @param[in] t Time.
  /// @param[in] x State.
  /// @param[in] u Concatenatin of the control input and Lagrange multiplier with respect to the equality constraints. 
  /// @param[in] lmd Costate. 
  /// @param[out] hx Evaluated value of the partial derivative of the Hamiltonian.
  /// @remark This method is intended to be used inside of the cgmres solvers and does not check size of each argument. 
  /// Use the overloaded method if you call this outside of the cgmres solvers. 
  ///
  void eval_hx(const double t, const double* x, const double* u, 
               const double* lmd, double* hx) const {
    const double x0 = -2*x[0];
    const double x1 = u[4]*(2*X_1 + x0) + u[5]*(2*X_2 + x0);
    const double x2 = 2*x[1];
    const double x3 = -x2;
    const double x4 = u[4]*(2*Y_1 + x3) + u[5]*(2*Y_2 + x3);
    const double x5 = u[0]*sin(x[2]);
    const double x6 = cos(x[2]);
    const double x7 = u[2]*sin(x[5]);
    const double x8 = cos(x[5]);
    const double x9 = x[0] - x[3];
    const double x10 = pow(x9, 2);
    const double x11 = (1.0/2.0)*q[3];
    const double x12 = sqrt(x10)*x11;
    const double x13 = x2 - 2*x[4];
    hx[0] = (1.0/2.0)*q[0]*(-2*t*vx_ref - x0) + x1;
    hx[1] = q[1]*x[1] + x4;
    hx[2] = -lmd[0]*x5 + lmd[1]*u[0]*x6 + q[2]*x[2] - r[0]*x5*(u[0]*x6 - vx_ref);
    hx[3] = 0;
    hx[4] = 0;
    hx[5] = -lmd[3]*x7 + lmd[4]*u[2]*x8;
    hx[6] = x1 + x12/x9;
    hx[7] = x11*x13 + x4;
    hx[8] = -lmd[6]*x5 + lmd[7]*u[0]*x6;
    hx[9] = -x12*x9/x10;
    hx[10] = -x11*x13;
    hx[11] = lmd[10]*u[2]*x8 - lmd[9]*x7 + q[4]*x[5];
 
  }

  ///
  /// @brief Computes the partial derivative of the Hamiltonian with respect to control input and the equality constraints, 
  /// i.e., hu = dH/du(t, x, u, lmd).
  /// @param[in] t Time.
  /// @param[in] x State.
  /// @param[in] u Concatenatin of the control input and Lagrange multiplier with respect to the equality constraints. 
  /// @param[in] lmd Costate. 
  /// @param[out] hu Evaluated value of the partial derivative of the Hamiltonian.
  /// @remark This method is intended to be used inside of the cgmres solvers and does not check size of each argument. 
  /// Use the overloaded method if you call this outside of the cgmres solvers. 
  ///
  void eval_hu(const double t, const double* x, const double* u, 
               const double* lmd, double* hu) const {
    const double x0 = cos(x[2]);
    const double x1 = -x[0];
    const double x2 = -x[1];
    const double x3 = -pow(R_1, 2) + pow(-X_1 - x1, 2) + pow(-Y_1 - x2, 2);
    const double x4 = -pow(R_2, 2) + pow(-X_2 - x1, 2) + pow(-Y_2 - x2, 2);
    hu[0] = lmd[0]*x0 + lmd[1]*sin(x[2]) + r[0]*x0*(u[0]*x0 - vx_ref);
    hu[1] = lmd[2] + r[1]*u[1];
    hu[2] = lmd[10]*sin(x[5]) + lmd[9]*cos(x[5]) + r[0]*u[2];
    hu[3] = lmd[11] + r[1]*u[3];
    hu[4] = -u[4] - x3 + sqrt(fb_eps[0] + pow(u[4], 2) + pow(x3, 2));
    hu[5] = -u[5] - x4 + sqrt(fb_eps[1] + pow(u[5], 2) + pow(x4, 2));
 
  }

  ///
  /// @brief Computes the state equation dx = f(t, x, u).
  /// @param[in] t Time.
  /// @param[in] x State. Size must be nx.
  /// @param[in] u Control input. Size must be nu.
  /// @param[out] dx Evaluated value of the state equation. Size must be nx.
  ///
  template <typename VectorType1, typename VectorType2, typename VectorType3>
  void eval_f(const double t, const MatrixBase<VectorType1>& x, 
              const MatrixBase<VectorType2>& u, 
              const MatrixBase<VectorType3>& dx) const {
    if (x.size() != nx) {
      throw std::invalid_argument("[OCP]: x.size() must be " + std::to_string(nx));
    }
    if (u.size() != nu) {
      throw std::invalid_argument("[OCP]: u.size() must be " + std::to_string(nu));
    }
    if (dx.size() != nx) {
      throw std::invalid_argument("[OCP]: dx.size() must be " + std::to_string(nx));
    }
    eval_f(t, x.derived().data(), u.derived().data(), CGMRES_EIGEN_CONST_CAST(VectorType3, dx).data());
  }

  ///
  /// @brief Computes the partial derivative of terminal cost with respect to state, 
  /// i.e., phix = dphi/dx(t, x).
  /// @param[in] t Time.
  /// @param[in] x State. Size must be nx.
  /// @param[out] phix Evaluated value of the partial derivative of terminal cost. Size must be nx.
  ///
  template <typename VectorType1, typename VectorType2>
  void eval_phix(const double t, const MatrixBase<VectorType1>& x, 
                 const MatrixBase<VectorType2>& phix) const {
    if (x.size() != nx) {
      throw std::invalid_argument("[OCP]: x.size() must be " + std::to_string(nx));
    }
    if (phix.size() != nx) {
      throw std::invalid_argument("[OCP]: phix.size() must be " + std::to_string(nx));
    }
    eval_phix(t, x.derived().data(), CGMRES_EIGEN_CONST_CAST(VectorType2, phix).data());
  }

  ///
  /// @brief Computes the partial derivative of the Hamiltonian with respect to the state, 
  /// i.e., hx = dH/dx(t, x, u, lmd).
  /// @param[in] t Time.
  /// @param[in] x State. Size must be nx.
  /// @param[in] uc Concatenatin of the control input and Lagrange multiplier with respect to the equality constraints. Size must be nuc. 
  /// @param[in] lmd Costate.  Size must be nx.
  /// @param[out] hx Evaluated value of the partial derivative of the Hamiltonian. Size must be nx.
  ///
  template <typename VectorType1, typename VectorType2, typename VectorType3, typename VectorType4>
  void eval_hx(const double t, const MatrixBase<VectorType1>& x, 
               const MatrixBase<VectorType2>& uc, 
               const MatrixBase<VectorType3>& lmd, 
               const MatrixBase<VectorType4>& hx) const {
    if (x.size() != nx) {
      throw std::invalid_argument("[OCP]: x.size() must be " + std::to_string(nx));
    }
    if (uc.size() != nuc) {
      throw std::invalid_argument("[OCP]: uc.size() must be " + std::to_string(nuc));
    }
    if (lmd.size() != nx) {
      throw std::invalid_argument("[OCP]: lmd.size() must be " + std::to_string(nx));
    }
    if (hx.size() != nuc) {
      throw std::invalid_argument("[OCP]: hx.size() must be " + std::to_string(nx));
    }
    eval_hx(t, x.derived().data(), uc.derived().data(), lmd.derived().data(), CGMRES_EIGEN_CONST_CAST(VectorType4, hx).data());
  }

  ///
  /// @brief Computes the partial derivative of the Hamiltonian with respect to control input and the equality constraints, 
  /// i.e., hu = dH/du(t, x, u, lmd).
  /// @param[in] t Time.
  /// @param[in] x State. Size must be nx.
  /// @param[in] uc Concatenatin of the control input and Lagrange multiplier with respect to the equality constraints. Size must be nuc. 
  /// @param[in] lmd Costate. Size must be nx. 
  /// @param[out] hu Evaluated value of the partial derivative of the Hamiltonian. Size must be nuc.
  ///
  template <typename VectorType1, typename VectorType2, typename VectorType3, typename VectorType4>
  void eval_hu(const double t, const MatrixBase<VectorType1>& x, 
               const MatrixBase<VectorType2>& uc, 
               const MatrixBase<VectorType3>& lmd, 
               const MatrixBase<VectorType4>& hu) const {
    if (x.size() != nx) {
      throw std::invalid_argument("[OCP]: x.size() must be " + std::to_string(nx));
    }
    if (uc.size() != nuc) {
      throw std::invalid_argument("[OCP]: uc.size() must be " + std::to_string(nuc));
    }
    if (lmd.size() != nx) {
      throw std::invalid_argument("[OCP]: lmd.size() must be " + std::to_string(nx));
    }
    if (hu.size() != nuc) {
      throw std::invalid_argument("[OCP]: hu.size() must be " + std::to_string(nuc));
    }
    eval_hu(t, x.derived().data(), uc.derived().data(), lmd.derived().data(), CGMRES_EIGEN_CONST_CAST(VectorType4, hu).data());
  }

};

} // namespace cgmres

#endif // CGMRES_OCP_HPP_
