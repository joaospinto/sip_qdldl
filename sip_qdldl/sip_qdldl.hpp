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
};

// Provides the callbacks required by SIP, given a set of sparsity patterns.
class CallbackProvider {
public:
  CallbackProvider(ConstSparseMatrix upper_H_pattern,
                   ConstSparseMatrix C_pattern, ConstSparseMatrix G_pattern,
                   const Settings &settings, Workspace &workspace);

  void ldlt_factor(const double *upper_H_data, const double *C_data,
                   const double *G_data, const double *w, const double r1,
                   const double r2, const double r3, double *LT_data,
                   double *D_diag);
  void ldlt_solve(const double *LT_data, const double *D_diag, const double *b,
                  double *v);
  void add_Kx_to_y(const double *upper_H_data, const double *C_data,
                   const double *G_data, const double *w, const double r1,
                   const double r2, const double r3, const double *x_x,
                   const double *x_y, const double *x_z, double *y_x,
                   double *y_y, double *y_z);
  void add_upper_symmetric_Hx_to_y(const double *upper_H_data, const double *x,
                                   double *y);
  void add_Cx_to_y(const double *C_data, const double *x, double *y);
  void add_CTx_to_y(const double *C_data, const double *x, double *y);
  void add_Gx_to_y(const double *G_data, const double *x, double *y);
  void add_GTx_to_y(const double *G_data, const double *x, double *y);

private:
  // Builds the upper-triangle representation of the 3x3 KKT LHS:
  //  K = [[ H + r1 I_x     C.T         G.T     ]
  //       [     C        -r2 I_y        0      ]
  //       [     G           0      -W - r3 I_z ]]
  // Note that W is a diagonal matrix, and that r1, r2, r3
  // are non-negative scalars.
  void build_lhs(const double *upper_H_data, const double *C_data,
                 const double *G_data, const double *w, const double r1,
                 const double r2, const double r3);

  int get_x_dim() const;
  int get_y_dim() const;
  int get_z_dim() const;
  int get_kkt_dim() const;

  const ConstSparseMatrix upper_H_pattern_;
  const ConstSparseMatrix C_pattern_;
  const ConstSparseMatrix G_pattern_;

  const Settings &settings_;
  Workspace &workspace_;
};

} // namespace sip_qdldl
