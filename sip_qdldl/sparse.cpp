#include "sparse.hpp"

#include <algorithm>
#include <cassert>

namespace sip_qdldl {

auto SparseMatrix::reserve(int dim, int nnz) -> void {
  ind = new int[nnz];
  indptr = new int[dim + 1];
  data = new double[nnz];
}

auto SparseMatrix::free() -> void {
  delete[] ind;
  delete[] indptr;
  delete[] data;
}

auto SparseMatrix::mem_assign(int dim, int nnz, unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  ind = reinterpret_cast<decltype(ind)>(mem_ptr + cum_size);
  cum_size += nnz * sizeof(int);
  indptr = reinterpret_cast<decltype(indptr)>(mem_ptr + cum_size);
  cum_size += (dim + 1) * sizeof(int);
  data = reinterpret_cast<decltype(data)>(mem_ptr + cum_size);
  cum_size += nnz * sizeof(double);
  return cum_size;
}

auto operator<<(std::ostream &os, const ConstSparseMatrix &M)
    -> std::ostream & {
  os << "rows: " << M.rows;
  os << "\ncols: " << M.cols;
  os << "\nindptr: ";
  for (int i = 0; i <= M.cols; ++i) {
    os << M.indptr[i];
    if (i < M.cols) {
      os << ", ";
    }
  }
  const int nnz = M.indptr[M.cols];
  os << "\nind: ";
  for (int i = 0; i < nnz; ++i) {
    os << M.ind[i];
    if (i + 1 < nnz) {
      os << ", ";
    }
  }
  os << "\ndata: ";
  for (int i = 0; i < nnz; ++i) {
    os << M.data[i];
    if (i + 1 < nnz) {
      os << ", ";
    }
  }
  os << "\nis_transposed: " << M.is_transposed;
  return os;
}

auto _add_ATx_to_y_impl(const ConstSparseMatrix &A, const double *x,
                        double *y) {
  for (int j = 0; j < A.cols; ++j) {
    for (int i = A.indptr[j]; i < A.indptr[j + 1]; ++i) {
      y[j] += A.data[i] * x[A.ind[i]];
    }
  }
}

auto _add_Ax_to_y_impl(const ConstSparseMatrix &A, const double *x, double *y)
    -> void {
  for (int j = 0; j < A.cols; j++) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      y[i] += A.data[value_idx] * x[j];
    }
  }
}

auto add_ATx_to_y(const ConstSparseMatrix &A, const double *x, double *y)
    -> void {
  if (A.is_transposed) {
    _add_Ax_to_y_impl(A, x, y);
  } else {
    _add_ATx_to_y_impl(A, x, y);
  }
}

auto add_Ax_to_y(const ConstSparseMatrix &A, const double *x, double *y)
    -> void {
  if (A.is_transposed) {
    _add_ATx_to_y_impl(A, x, y);
  } else {
    _add_Ax_to_y_impl(A, x, y);
  }
}

auto add_Ax_to_y_where_A_upper_symmetric(const ConstSparseMatrix &A,
                                         const double *x, double *y) -> void {
  _add_ATx_to_y_impl(A, x, y);
  _add_Ax_to_y_impl(A, x, y);
  for (int j = 0; j < A.cols; ++j) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      if (i == j) {
        y[i] -= A.data[value_idx] * x[j];
      }
    }
  }
}

auto csc_cumsum(int *p, int *c, const int n) -> int {
  // https://github.com/osqp/osqp/blob/4532d356f08789461bc041531f22a1001144c40a/algebra/_common/csc_utils.c#L68
  int nz = 0;
  for (int i = 0; i < n; i++) {
    p[i] = nz;
    nz += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz;
}

auto permute(const ConstSparseMatrix &A, const int *pinv,
             int *permutation_workspace, int *AtoC, SparseMatrix &C) -> void {
  // https://github.com/osqp/osqp/blob/4532d356f08789461bc041531f22a1001144c40a/algebra/_common/csc_utils.c#L326
  assert(A.rows == A.cols);
  const int n = A.rows;
  const int *Ap = A.indptr;
  const int *Ai = A.ind;
  const double *Ax = A.data;

  C.rows = n;
  C.cols = n;
  C.is_transposed = A.is_transposed;

  int *Cp = C.indptr;
  int *Ci = C.ind;
  double *Cx = C.data;

  int *w = permutation_workspace;
  std::fill(w, w + n, 0);

  for (int j = 0; j < n; j++) {
    const int j2 = pinv ? pinv[j] : j;

    for (int p = Ap[j]; p < Ap[j + 1]; p++) {
      const int i = Ai[p];

      if (i > j)
        continue;
      const int i2 = pinv ? pinv[i] : i;
      w[std::max(i2, j2)]++;
    }
  }

  csc_cumsum(Cp, w, n);

  for (int j = 0; j < n; j++) {
    const int j2 = pinv ? pinv[j] : j;

    for (int p = Ap[j]; p < Ap[j + 1]; p++) {
      const int i = Ai[p];

      if (i > j)
        continue;
      const int i2 = pinv ? pinv[i] : i;
      const int q = w[std::max(i2, j2)]++;
      Ci[q] = std::min(i2, j2);

      if (Cx)
        Cx[q] = Ax[p];

      if (AtoC) {
        AtoC[p] = q;
      }
    }
  }
}

} // namespace sip_qdldl
