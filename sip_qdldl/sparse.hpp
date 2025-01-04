#pragma once

#include <ostream>

namespace sip_qdldl {

struct SparseMatrix {
  // The number of rows of the matrix.
  int rows;
  // The number of cols of the matrix.
  int cols;
  // The row indices of each entry.
  int *ind;
  // The column start indices of each column.
  int *indptr;
  // The potentially non-zero entries.
  double *data;
  // Whether the matrix is transposed.
  bool is_transposed;

  // To dynamically allocate the required memory.
  auto reserve(int dim, int nnz) -> void;
  auto free() -> void;

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int dim, int nnz, unsigned char *mem_ptr) -> int;
};

struct ConstSparseMatrix {
  // The number of rows of the matrix.
  const int rows;
  // The number of cols of the matrix.
  const int cols;
  // The row indices of each entry.
  const int *const ind;
  // The column start indices of each column.
  const int *const indptr;
  // The potentially non-zero entries.
  const double *const data;
  // Whether the matrix is transposed.
  const bool is_transposed;

  ConstSparseMatrix(const int _rows, const int _cols, const int *const _ind,
                    const int *const _indptr, const double *const _data,
                    const bool _is_transposed)
      : rows(_rows), cols(_cols), ind(_ind), indptr(_indptr), data(_data),
        is_transposed(_is_transposed) {}

  ConstSparseMatrix(const SparseMatrix &M)
      : rows(M.rows), cols(M.cols), ind(M.ind), indptr(M.indptr), data(M.data),
        is_transposed(M.is_transposed) {}

  ConstSparseMatrix(const ConstSparseMatrix &M, const double *data)
      : rows(M.rows), cols(M.cols), ind(M.ind), indptr(M.indptr), data(data),
        is_transposed(M.is_transposed) {}
};

// Useful for debugging.
auto operator<<(std::ostream &os, const ConstSparseMatrix &M) -> std::ostream &;

auto add_ATx_to_y(const ConstSparseMatrix &A, const double *x, double *y)
    -> void;

auto add_Ax_to_y(const ConstSparseMatrix &A, const double *x, double *y)
    -> void;

auto add_Ax_to_y_where_A_upper_symmetric(const ConstSparseMatrix &A,
                                         const double *x, double *y) -> void;

// NOTE: permutation_workspace should have size dim(A); AtoC is optional.
auto permute(const ConstSparseMatrix &A, const int *pinv,
             int *permutation_workspace, int *AtoC, SparseMatrix &C) -> void;

} // namespace sip_qdldl
