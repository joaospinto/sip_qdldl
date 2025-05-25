#include "sip_qdldl.hpp"

#include <algorithm>
#include <cassert>

#include <qdldl.h>

namespace sip_qdldl {

CallbackProvider::CallbackProvider(const Settings &settings,
                                   ModelCallbackOutput &mco,
                                   Workspace &workspace)
    : settings_(settings), mco_(mco), workspace_(workspace) {
  assert(mco_.jacobian_c.is_transposed);
  assert(mco_.jacobian_g.is_transposed);
}

int CallbackProvider::get_x_dim() const {
  return mco_.upper_hessian_lagrangian.rows;
}

int CallbackProvider::get_y_dim() const { return mco_.jacobian_c.cols; }

int CallbackProvider::get_z_dim() const { return mco_.jacobian_g.cols; }

int CallbackProvider::get_kkt_dim() const {
  return get_x_dim() + get_y_dim() + get_z_dim();
}

void CallbackProvider::build_lhs(const double *w, const double r1,
                                 const double r2, const double r3) {
  // This function builds the following matrix in CSC format:
  // [ H + r1 I     C.T          G.T    ]
  // [    0      -r2 * I_y        0     ]
  // [    0          0        -W - r3 I ]
  const int x_dim = get_x_dim();
  const int y_dim = get_y_dim();
  const int z_dim = get_z_dim();

  SparseMatrix &lhs = workspace_.kkt_workspace.lhs;

  lhs.rows = get_kkt_dim();
  lhs.cols = lhs.rows;
  lhs.is_transposed = false;

  int k = 0;

  lhs.indptr[0] = k;

  // Fill the first block-column (H + r1 I).
  for (int i = 0; i < x_dim; ++i) {
    bool added_r1_term = false;
    for (int l = mco_.upper_hessian_lagrangian.indptr[i];
         l < mco_.upper_hessian_lagrangian.indptr[i + 1]; ++l) {
      const int j = mco_.upper_hessian_lagrangian.ind[l];
      if (j <= i) {
        lhs.ind[k] = mco_.upper_hessian_lagrangian.ind[l];
        lhs.data[k] = mco_.upper_hessian_lagrangian.data[l];
        if (i == mco_.upper_hessian_lagrangian.ind[l]) {
          lhs.data[k] += r1;
          added_r1_term = true;
        }
        ++k;
      }
    }
    if (!added_r1_term && r1 > 0.0) {
      lhs.ind[k] = i;
      lhs.data[k] = r1;
      ++k;
    }
    lhs.indptr[i + 1] = k;
  }

  // Fill the second block column (C.T and -r2 * I_y).
  for (int i = 0; i < y_dim; ++i) {
    // Fill C.T column.
    // NOTE: assumes that mco_.jacobian_c.is_transposed == true.
    for (int j = mco_.jacobian_c.indptr[i]; j < mco_.jacobian_c.indptr[i + 1];
         ++j) {
      lhs.ind[k] = mco_.jacobian_c.ind[j];
      lhs.data[k] = mco_.jacobian_c.data[j];
      ++k;
    }
    // Fill -r2 * I_y column.
    const int row = x_dim + i;
    lhs.ind[k] = row;
    lhs.data[k] = -r2;
    ++k;
    lhs.indptr[row + 1] = k;
  }

  // Fill the third block-column (G.T and -W - r3 I).
  for (int i = 0; i < z_dim; ++i) {
    // Fill G.T column.
    // NOTE: assumes that mco_.jacobian_g.is_transposed == true.
    for (int j = mco_.jacobian_g.indptr[i]; j < mco_.jacobian_g.indptr[i + 1];
         ++j) {
      lhs.ind[k] = mco_.jacobian_g.ind[j];
      lhs.data[k] = mco_.jacobian_g.data[j];
      ++k;
    }
    // Fill -W - r3 I column.
    const int row = x_dim + y_dim + i;
    lhs.ind[k] = row;
    lhs.data[k] = -w[i] - r3;
    ++k;
    lhs.indptr[row + 1] = k;
  }
}

void CallbackProvider::factor(const double *w, const double r1, const double r2,
                              const double r3) {
  build_lhs(w, r1, r2, r3);

  if (settings_.permute_kkt_system) {
    assert(settings_.kkt_pinv != nullptr);
    int *AtoC = nullptr;
    permute(workspace_.kkt_workspace.lhs, settings_.kkt_pinv,
            workspace_.permutation_workspace, AtoC,
            workspace_.kkt_workspace.permuted_lhs);
  }

  const auto &lhs = settings_.permute_kkt_system
                        ? workspace_.kkt_workspace.permuted_lhs
                        : workspace_.kkt_workspace.lhs;

  const int kkt_dim = get_kkt_dim();

  [[maybe_unused]] const int sumLnz = QDLDL_etree(
      lhs.rows, lhs.indptr, lhs.ind, workspace_.qdldl_workspace.iwork,
      workspace_.qdldl_workspace.Lnz, workspace_.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  [[maybe_unused]] const int num_pos_D_entries = QDLDL_factor(
      kkt_dim, lhs.indptr, lhs.ind, lhs.data, workspace_.qdldl_workspace.Lp,
      workspace_.qdldl_workspace.Li, workspace_.qdldl_workspace.Lx,
      workspace_.qdldl_workspace.D, workspace_.qdldl_workspace.Dinv,
      workspace_.qdldl_workspace.Lnz, workspace_.qdldl_workspace.etree,
      workspace_.qdldl_workspace.bwork, workspace_.qdldl_workspace.iwork,
      workspace_.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);
}

void CallbackProvider::solve(const double *b, double *v) {
  const int kkt_dim = get_kkt_dim();

  double *solution =
      settings_.permute_kkt_system ? workspace_.qdldl_workspace.x : v;

  if (settings_.permute_kkt_system) {
    for (int i = 0; i < kkt_dim; ++i) {
      solution[settings_.kkt_pinv[i]] = b[i];
    }
  } else {
    std::copy(b, b + kkt_dim, solution);
  }

  QDLDL_solve(kkt_dim, workspace_.qdldl_workspace.Lp,
              workspace_.qdldl_workspace.Li, workspace_.qdldl_workspace.Lx,
              workspace_.qdldl_workspace.Dinv, solution);

  if (settings_.permute_kkt_system) {
    for (int i = 0; i < kkt_dim; ++i) {
      v[i] = workspace_.qdldl_workspace.x[settings_.kkt_pinv[i]];
    }
  }
}

void CallbackProvider::add_Kx_to_y(const double *w, const double r1,
                                   const double r2, const double r3,
                                   const double *x_x, const double *x_y,
                                   const double *x_z, double *y_x, double *y_y,
                                   double *y_z) {
  add_Hx_to_y(x_x, y_x);
  add_Cx_to_y(x_x, y_y);
  add_CTx_to_y(x_y, y_x);
  add_Gx_to_y(x_x, y_z);
  add_GTx_to_y(x_z, y_x);
  const int x_dim = get_x_dim();
  const int y_dim = get_y_dim();
  const int z_dim = get_z_dim();
  for (int i = 0; i < x_dim; ++i) {
    y_x[i] += r1 * x_x[i];
  }
  for (int i = 0; i < y_dim; ++i) {
    y_y[i] -= r2 * x_y[i];
  }
  for (int i = 0; i < z_dim; ++i) {
    y_z[i] -= (w[i] + r3) * x_z[i];
  }
}

void CallbackProvider::add_Hx_to_y(const double *x, double *y) {
  add_Ax_to_y_where_A_upper_symmetric(mco_.upper_hessian_lagrangian, x, y);
}

void CallbackProvider::add_Cx_to_y(const double *x, double *y) {
  add_Ax_to_y(mco_.jacobian_c, x, y);
}

void CallbackProvider::add_CTx_to_y(const double *x, double *y) {
  add_ATx_to_y(mco_.jacobian_c, x, y);
}

void CallbackProvider::add_Gx_to_y(const double *x, double *y) {
  add_Ax_to_y(mco_.jacobian_g, x, y);
}

void CallbackProvider::add_GTx_to_y(const double *x, double *y) {
  add_ATx_to_y(mco_.jacobian_g, x, y);
}

void QDLDLWorkspace::reserve(int kkt_dim, int kkt_L_nnz) {
  etree = new int[kkt_dim];
  Lnz = new int[kkt_dim];
  iwork = new int[3 * kkt_dim];
  bwork = new unsigned char[kkt_dim];
  fwork = new double[kkt_dim];
  Lp = new int[kkt_dim + 1];
  Li = new int[kkt_L_nnz];
  Lx = new double[kkt_L_nnz];
  D = new double[kkt_dim];
  Dinv = new double[kkt_dim];
  x = new double[kkt_dim];
}

void QDLDLWorkspace::free() {
  delete[] etree;
  delete[] Lnz;
  delete[] iwork;
  delete[] bwork;
  delete[] fwork;
  delete[] Lp;
  delete[] Li;
  delete[] Lx;
  delete[] D;
  delete[] Dinv;
  delete[] x;
}

auto QDLDLWorkspace::mem_assign(int kkt_dim, int kkt_L_nnz,
                                unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  etree = reinterpret_cast<decltype(etree)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  Lnz = reinterpret_cast<decltype(Lnz)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  iwork = reinterpret_cast<decltype(iwork)>(mem_ptr + cum_size);
  cum_size += 3 * kkt_dim * sizeof(int);

  bwork = reinterpret_cast<decltype(bwork)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(unsigned char);

  fwork = reinterpret_cast<decltype(fwork)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  Lp = reinterpret_cast<decltype(Lp)>(mem_ptr + cum_size);
  cum_size += (kkt_dim + 1) * sizeof(int);

  Li = reinterpret_cast<decltype(Li)>(mem_ptr + cum_size);
  cum_size += kkt_L_nnz * sizeof(int);

  Lx = reinterpret_cast<decltype(Lx)>(mem_ptr + cum_size);
  cum_size += kkt_L_nnz * sizeof(double);

  D = reinterpret_cast<decltype(D)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  Dinv = reinterpret_cast<decltype(Dinv)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  x = reinterpret_cast<decltype(x)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  return cum_size;
}

void KKTWorkspace::reserve(int kkt_dim, int kkt_nnz) {
  lhs.reserve(kkt_dim, kkt_nnz);
  permuted_lhs.reserve(kkt_dim, kkt_nnz);
}

void KKTWorkspace::free() {
  lhs.free();
  permuted_lhs.free();
}

auto KKTWorkspace::mem_assign(int kkt_dim, int kkt_nnz, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  cum_size += lhs.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);
  cum_size += permuted_lhs.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);

  return cum_size;
}

void Workspace::reserve(int kkt_dim, int kkt_nnz, int kkt_L_nnz) {
  kkt_workspace.reserve(kkt_dim, kkt_nnz);
  qdldl_workspace.reserve(kkt_dim, kkt_L_nnz);
  permutation_workspace = new int[kkt_dim];
}

void Workspace::free() {
  kkt_workspace.free();
  qdldl_workspace.free();
  delete[] permutation_workspace;
}

auto Workspace::mem_assign(int kkt_dim, int kkt_nnz, int kkt_L_nnz,
                           unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  cum_size += kkt_workspace.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);
  cum_size +=
      qdldl_workspace.mem_assign(kkt_dim, kkt_L_nnz, mem_ptr + cum_size);
  permutation_workspace =
      reinterpret_cast<decltype(permutation_workspace)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  return cum_size;
}

void ModelCallbackOutput::reserve(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_lagrangian_nnz,
                                  int jacobian_c_nnz, int jacobian_g_nnz,
                                  bool is_jacobian_c_transposed,
                                  bool is_jacobian_g_transposed) {
  gradient_f = new double[x_dim];
  upper_hessian_lagrangian.reserve(x_dim, upper_hessian_lagrangian_nnz);
  c = new double[y_dim];
  if (is_jacobian_c_transposed) {
    jacobian_c.reserve(y_dim, jacobian_c_nnz);
  } else {
    jacobian_c.reserve(x_dim, jacobian_c_nnz);
  }
  g = new double[s_dim];
  if (is_jacobian_g_transposed) {
    jacobian_g.reserve(s_dim, jacobian_g_nnz);
  } else {
    jacobian_g.reserve(x_dim, jacobian_g_nnz);
  }
}

void ModelCallbackOutput::free() {
  delete[] gradient_f;
  upper_hessian_lagrangian.free();
  delete[] c;
  jacobian_c.free();
  delete[] g;
  jacobian_g.free();
}

auto ModelCallbackOutput::mem_assign(int x_dim, int s_dim, int y_dim,
                                     int upper_hessian_lagrangian_nnz,
                                     int jacobian_c_nnz, int jacobian_g_nnz,
                                     bool is_jacobian_c_transposed,
                                     bool is_jacobian_g_transposed,
                                     unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  gradient_f = reinterpret_cast<decltype(gradient_f)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  cum_size += upper_hessian_lagrangian.mem_assign(
      x_dim, upper_hessian_lagrangian_nnz, mem_ptr + cum_size);

  c = reinterpret_cast<decltype(c)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  if (is_jacobian_c_transposed) {
    cum_size +=
        jacobian_c.mem_assign(y_dim, jacobian_c_nnz, mem_ptr + cum_size);
  } else {
    cum_size +=
        jacobian_c.mem_assign(x_dim, jacobian_c_nnz, mem_ptr + cum_size);
  }

  g = reinterpret_cast<decltype(g)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  if (is_jacobian_g_transposed) {
    cum_size +=
        jacobian_g.mem_assign(s_dim, jacobian_g_nnz, mem_ptr + cum_size);
  } else {
    cum_size +=
        jacobian_g.mem_assign(x_dim, jacobian_g_nnz, mem_ptr + cum_size);
  }

  return cum_size;
}

} // namespace sip_qdldl
