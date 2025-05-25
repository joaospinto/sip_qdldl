#pragma once

#include "sparse.hpp"

namespace sip_qdldl {

struct Settings {
  // Whether to apply a permutation to the KKT system to reduce fill-in.
  bool permute_kkt_system = false;
  // A permutation for reducing fill-in in the KKT system.
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

  // Factorization output storage.
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

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int kkt_dim, int kkt_L_nnz) -> int {
    return (7 * kkt_dim + 1) * sizeof(int) + kkt_dim * sizeof(unsigned char) +
           (4 * kkt_dim + kkt_L_nnz) * sizeof(double);
  }
};

struct KKTWorkspace {
  // The LHS of the (potentially reduced/eliminated) KKT system.
  SparseMatrix lhs;
  // The permuted LHS (to avoid fill-in).
  SparseMatrix permuted_lhs;

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_nnz, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int kkt_dim, int kkt_nnz) -> int {
    return SparseMatrix::num_bytes(kkt_dim, kkt_nnz) +
           SparseMatrix::num_bytes(kkt_dim, kkt_nnz);
  }
};

// This data structure is used to avoid doing dynamic memory allocation inside
// of the solver, as well as avoiding excessive templating in the solver code.
struct Workspace {
  // Storage of the LHS and RHS of the KKT system.
  KKTWorkspace kkt_workspace;
  // The workspace of the QDLDL solver.
  QDLDLWorkspace qdldl_workspace;
  // Stores the permutation workspace.
  int *permutation_workspace;

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_nnz, int kkt_L_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_nnz, int kkt_L_nnz,
                  unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int kkt_dim, int kkt_nnz, int kkt_L_nnz)
      -> int {
    return KKTWorkspace::num_bytes(kkt_dim, kkt_nnz) +
           QDLDLWorkspace::num_bytes(kkt_dim, kkt_L_nnz) +
           kkt_dim * sizeof(int);
  }
};

struct ModelCallbackOutput {
  // NOTE: all sparse matrices should be represented in CSC format.

  // The objective and its first derivative.
  double f;
  double *gradient_f;

  // The Hessian of the Lagrangian.
  // NOTE:
  // 1. Only the upper triangle should be filled in upper_hessian_lagrangian.
  // 2. upper_hessian_lagrangian should be a positive definite approximation.
  // 3. An positive definite approximation of the Hessian of f is often used.
  SparseMatrix upper_hessian_lagrangian;

  // The equality constraints and their first derivative.
  double *c;
  SparseMatrix jacobian_c;

  // The inequality constraints and their first derivative.
  double *g;
  SparseMatrix jacobian_g;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim,
               int upper_hessian_lagrangian_nnz, int jacobian_c_nnz,
               int jacobian_g_nnz, bool is_jacobian_c_transposed,
               bool is_jacobian_g_transposed);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim,
                  int upper_hessian_lagrangian_nnz, int jacobian_c_nnz,
                  int jacobian_g_nnz, bool is_jacobian_c_transposed,
                  bool is_jacobian_g_transposed, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_lagrangian_nnz,
                                  int jacobian_c_nnz, int jacobian_g_nnz,
                                  bool is_jacobian_c_transposed,
                                  bool is_jacobian_g_transposed) -> int {
    int out = (x_dim + s_dim + y_dim) * sizeof(double) +
              SparseMatrix::num_bytes(x_dim, upper_hessian_lagrangian_nnz);
    if (is_jacobian_c_transposed) {
      out += SparseMatrix::num_bytes(y_dim, jacobian_c_nnz);
    } else {
      out += SparseMatrix::num_bytes(x_dim, jacobian_c_nnz);
    }
    if (is_jacobian_g_transposed) {
      out += SparseMatrix::num_bytes(s_dim, jacobian_g_nnz);
    } else {
      out += SparseMatrix::num_bytes(x_dim, jacobian_g_nnz);
    }
    return out;
  }
};

// Provides the callbacks required by SIP, given a set of sparsity patterns.
class CallbackProvider {
public:
  CallbackProvider(const Settings &settings, ModelCallbackOutput &mco,
                   Workspace &workspace);

  void factor(const double *w, const double r1, const double r2,
              const double r3);
  void solve(const double *b, double *v);
  void add_Kx_to_y(const double *w, const double r1, const double r2,
                   const double r3, const double *x_x, const double *x_y,
                   const double *x_z, double *y_x, double *y_y, double *y_z);
  void add_Hx_to_y(const double *x, double *y);
  void add_Cx_to_y(const double *x, double *y);
  void add_CTx_to_y(const double *x, double *y);
  void add_Gx_to_y(const double *x, double *y);
  void add_GTx_to_y(const double *x, double *y);

private:
  // Builds the upper-triangle representation of the 3x3 KKT LHS:
  //  K = [[ H + r1 I_x     C.T         G.T     ]
  //       [     C        -r2 I_y        0      ]
  //       [     G           0      -W - r3 I_z ]]
  // Note that W is a diagonal matrix, and that r1, r2, r3
  // are non-negative scalars.
  void build_lhs(const double *w, const double r1, const double r2,
                 const double r3);

  int get_x_dim() const;
  int get_y_dim() const;
  int get_z_dim() const;
  int get_kkt_dim() const;

  const Settings &settings_;
  ModelCallbackOutput &mco_;
  Workspace &workspace_;
};

} // namespace sip_qdldl
