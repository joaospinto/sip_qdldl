#pragma once

#include <optional>
#include <ostream>

#include "sparse.hpp"

namespace sip_qdldl {

struct Input {
  // The equality constraint vector c(x).
  const double *c;
  // The inequality constraint vector g(x).
  const double *g;
  // The gradient of the cost f(x).
  const double *grad_f;
  // A positive-definite approximation of the Hessian of the cost f(x).
  // NOTE: only the upper half of H should be filled.
  ConstSparseMatrix H;
  // The Jacobian of c(x).
  ConstSparseMatrix C;
  // The Jacobian of g(x).
  ConstSparseMatrix G;
  // The current candidate slack variables.
  const double *s;
  // The current candidate equality multipliers.
  const double *y;
  // The current candidate inequality multipliers.
  const double *z;
  // The current candidate elastic variables.
  const double *e;
  // The barrier coefficient.
  const double mu;
  // The constant p so that elastic variable costs are 0.5 * p * ||e||^2.
  const double p;
  // The regularization term on the x variables.
  const double r1;
  // The regularization term on the y variables.
  const double r2;
  // The regularization term on the z variables.
  const double r3;
};

struct Output {
  // The proposed change to the primal x variables.
  double *dx;
  // The proposed change to the slacks.
  double *ds;
  // The proposed change to the equality multipliers.
  double *dy;
  // The proposed change to the inequality multipliers.
  double *dz;
  // The proposed change to the elastic variables.
  double *de;
  // The Newton-KKT error of the inputs.
  double kkt_error;
  // The error of this solution of the Newton-KKT system.
  double lin_sys_error;
};

struct Settings {
  // Determines how the Newton-KKT system is solved.
  enum class LinearSystemFormulation {
    SYMMETRIC_DIRECT_4x4 = 0,
    SYMMETRIC_INDIRECT_3x3 = 1,
    SYMMETRIC_INDIRECT_2x2 = 2,
  };
  // Determines how the search direction is computed.
  LinearSystemFormulation lin_sys_formulation =
      LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3;
  // Whether elastic variables are enabled.
  bool enable_elastics;
  // Whether to apply a permutation to the KKT system to reduce fill-in.
  bool permute_kkt_system = false;
  // A permutation for reducing fill-in in the KKT system.
  const int *kkt_p{nullptr};
  // The inverse of kkt_p.
  const int *kkt_pinv{nullptr};
};

struct QDLDLWorkspace {
  // Elimination tree workspace.
  int *etree; // Required size: kkt_dim
  int *Lnz;   // Required size: kkt_dim

  // Factorization workspace.
  int *iwork;           // Required size: 3 * kkt_dim
  unsigned char *bwork; // Required size: kkt_dim
  double *fwork;        // Required size: kkt_dim

  // Factorizaton output storage.
  int *Lp;      // Required size: kkt_dim + 1
  int *Li;      // Required size: kkt_L_nnz
  double *Lx;   // Required size: kkt_L_nnz
  double *D;    // Required size: kkt_dim
  double *Dinv; // Required size: kkt_dim

  // Solve workspace.
  double *x; // Required size: kkt_dim

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_L_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_L_nnz, unsigned char *mem_ptr) -> int;
};

struct MiscellaneousWorkspace {
  // Stores g(x) + s.
  double *g_plus_s;
  // Stores g(x) + s (+ e, when applicable).
  double *g_plus_s_plus_e;
  // Stores the linear system residual.
  double *lin_sys_residual;
  // Stores the x-gradient of the Lagrangian.
  double *grad_x_lagrangian;
  // Stores sigma = z / (s + gamma_z * z).
  double *sigma;
  // Stores sigma * (g(x) + (mu / z)).
  double *sigma_times_g_plus_mu_over_z_minus_z_over_p;
  // Stores jacobian_g_t @ sigma @ jacobian_g.
  SparseMatrix jac_g_t_sigma_jac_g;
  // Scratch space for the permutation method.
  int *permutation_workspace;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int kkt_dim, int jac_g_t_jac_g_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int kkt_dim, int jac_g_t_jac_g_nnz,
                  unsigned char *mem_ptr) -> int;
};

struct KKTWorkspace {
  // The LHS of the (potentially reduced/eliminated) KKT system.
  SparseMatrix lhs;
  // The permuted LHS (to avoid fill-in).
  SparseMatrix permuted_lhs;
  // The (negative) RHS of the (potentially reduced/eliminated )KKT system.
  double *negative_rhs;

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_nnz, unsigned char *mem_ptr) -> int;
};

// This data structure is used to avoid doing dynamic memory allocation inside
// of the solver, as well as avoiding excessive templating in the solver code.
struct Workspace {
  // Storage of the LHS and RHS of the KKT system.
  KKTWorkspace kkt_workspace;
  // The workspace of the QDLDL solver.
  QDLDLWorkspace qdldl_workspace;
  // Stores miscellaneous items.
  MiscellaneousWorkspace miscellaneous_workspace;

  // To dynamically allocate the required memory.
  void reserve(Settings::LinearSystemFormulation lin_sys_formulation, int x_dim,
               int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
               int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int kkt_L_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(Settings::LinearSystemFormulation lin_sys_formulation,
                  int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
                  int jacobian_c_nnz, int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
                  int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz,
                  int kkt_L_nnz, unsigned char *mem_ptr) -> int;
};

void compute_search_direction(const Input &input, const Settings &settings,
                              Workspace &workspace, Output &output);

} // namespace sip_qdldl
