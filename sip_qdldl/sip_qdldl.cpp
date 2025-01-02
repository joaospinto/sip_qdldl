#include "sip_qdldl.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include <qdldl.h>

namespace sip_qdldl {

auto get_s_dim(const ConstSparseMatrix &jacobian_g) -> int {
  return jacobian_g.is_transposed ? jacobian_g.cols : jacobian_g.rows;
}

auto get_y_dim(const ConstSparseMatrix &jacobian_c) -> int {
  return jacobian_c.is_transposed ? jacobian_c.cols : jacobian_c.rows;
}

void build_lhs_4x4(const Input &input, const Settings &settings,
                   Workspace &workspace) {
  // Builds the following matrix in CSC format:
  // [ H       0         C.T              G.T       ]
  // [ 0   S^{-1} Z       0               I_s       ]
  // [ 0       0      -r2 * I_y            0        ]
  // [ 0       0          0       -(r3 + 1/p) * I_z ]
  const int x_dim = input.H.rows;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  const double *s = input.s;
  const double *z = input.z;

  auto &lhs = workspace.kkt_workspace.lhs;

  lhs.rows = x_dim + 2 * s_dim + y_dim;
  lhs.cols = lhs.rows;
  lhs.is_transposed = false;

  int k = 0;

  // Fill upper_hessian_f.
  for (int i = 0; i < x_dim; ++i) {
    lhs.indptr[i] = k;
    for (int j = input.H.indptr[i]; j < input.H.indptr[i + 1]; ++j) {
      if (input.H.ind[j] <= i) {
        lhs.ind[k] = input.H.ind[j];
        lhs.data[k] = input.H.data[j];
        ++k;
      }
    }
  }

  // Fill S^{-1} Z.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + i] = k;
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = z[i] / s[i];
    ++k;
  }

  // Fill jacobian_c_t and -gamma_y * I_y.
  for (int i = 0; i < y_dim; ++i) {
    lhs.indptr[x_dim + s_dim + i] = k;
    // Fill jacobian_c_t column.
    // NOTE: input.C.is_transposed == true.
    for (int j = input.C.indptr[i]; j < input.C.indptr[i + 1]; ++j) {
      lhs.ind[k] = input.C.ind[j];
      lhs.data[k] = input.C.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + s_dim + i;
    lhs.data[k] = -input.r2;
    ++k;
  }

  // Fill jacobian_g, I_s, and -(gamma_z + 1/p) * I_z.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + s_dim + y_dim + i] = k;
    // Fill jacobian_g column.
    // NOTE: input.G.is_transposed == true.
    for (int j = input.G.indptr[i]; j < input.G.indptr[i + 1]; ++j) {
      lhs.ind[k] = input.G.ind[j];
      lhs.data[k] = input.G.data[j];
      ++k;
    }
    // Fill I_s column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = 1.0;
    ++k;
    // Fill -(gamma_z + 1 / p) * I_z column.
    lhs.ind[k] = x_dim + s_dim + y_dim + i;
    lhs.data[k] = -input.r3 - (settings.enable_elastics ? 1.0 / input.p : 0.0);
    ++k;
  }

  lhs.indptr[x_dim + y_dim + 2 * s_dim] = k;
}

void build_nrhs_4x4(const Input &input, const Settings &settings,
                    Workspace &workspace) {
  // Builds the following vector:
  // [ grad_f + C.T @ y + G.T @ z ]
  // [      z - input.mu / s      ]
  // [             c              ]
  // [       g + s - z / p        ]
  const double *s = input.s;
  const double *y = input.y;
  const double *z = input.z;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(input.grad_f, input.grad_f + x_dim, nrhs);

  add_ATx_to_y(input.C, y, nrhs);
  add_ATx_to_y(input.G, z, nrhs);

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + i] = z[i] - input.mu / s[i];
  }

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + s_dim + i] = input.c[i];
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      workspace.miscellaneous_workspace.g_plus_s[i] = input.g[i] + input.s[i];
      nrhs[x_dim + s_dim + y_dim + i] =
          workspace.miscellaneous_workspace.g_plus_s[i] - z[i] / input.p;
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      workspace.miscellaneous_workspace.g_plus_s[i] = input.g[i] + input.s[i];
      nrhs[x_dim + s_dim + y_dim + i] =
          workspace.miscellaneous_workspace.g_plus_s[i];
    }
  }
}

void compute_search_direction_4x4(const Input &input, const Settings &settings,
                                  Workspace &workspace, Output &output) {
  build_lhs_4x4(input, settings, workspace);
  build_nrhs_4x4(input, settings, workspace);

  if (settings.permute_kkt_system) {
    assert(settings.kkt_p != nullptr);
    assert(settings.kkt_pinv != nullptr);
    int *AtoC = nullptr;
    permute(workspace.kkt_workspace.lhs, settings.kkt_pinv,
            workspace.miscellaneous_workspace.permutation_workspace, AtoC,
            workspace.kkt_workspace.permuted_lhs);
  }

  const auto &lhs = settings.permute_kkt_system
                        ? workspace.kkt_workspace.permuted_lhs
                        : workspace.kkt_workspace.lhs;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  output.kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    output.kkt_error = std::max(
        output.kkt_error, std::fabs(workspace.kkt_workspace.negative_rhs[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    output.kkt_error = std::max(output.kkt_error, std::fabs(input.c[i]));
  }
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error =
          std::max(output.kkt_error,
                   std::fabs(workspace.miscellaneous_workspace.g_plus_s[i] +
                             input.e[i]));
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error =
          std::max(output.kkt_error,
                   std::fabs(workspace.miscellaneous_workspace.g_plus_s[i]));
    }
  }

  const int dim = x_dim + y_dim + 2 * s_dim;

  [[maybe_unused]] const int sumLnz = QDLDL_etree(
      lhs.rows, lhs.indptr, lhs.ind, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  [[maybe_unused]] const int num_pos_D_entries = QDLDL_factor(
      dim, lhs.indptr, lhs.ind, lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] =
          -workspace.kkt_workspace.negative_rhs[settings.kkt_p[i]];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  add_Ax_to_y_where_A_upper_symmetric(
      lhs, workspace.qdldl_workspace.x,
      workspace.miscellaneous_workspace.lin_sys_residual);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[i]];
    }

    int offset = x_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.ds[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[offset + i]];
    }

    offset += s_dim;

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[offset + i]];
    }

    offset += y_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.dz[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[offset + i]];
    }
  } else {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[i];
    }

    int offset = x_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.ds[i] = workspace.qdldl_workspace.x[offset + i];
    }

    offset += s_dim;

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[offset + i];
    }

    offset += y_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.dz[i] = workspace.qdldl_workspace.x[offset + i];
    }
  }

  // de = -e - (dz + z) / p
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      const double p = input.p;
      output.de[i] = -input.e[i] - (output.dz[i] + input.z[i]) / p;
    }
  }

  output.lin_sys_error = 0.0;

  for (int i = 0; i < dim; ++i) {
    output.lin_sys_error = std::max(
        output.lin_sys_error,
        std::fabs(workspace.miscellaneous_workspace.lin_sys_residual[i]));
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.lin_sys_error = std::max(
          std::fabs((output.dz[i] + output.de[i] + input.z[i]) / input.p +
                    input.e[i]),
          output.lin_sys_error);
    }
  }
}

void build_lhs_3x3(const Input &input, const Settings &settings,
                   Workspace &workspace) {
  // Builds the following matrix in CSC format:
  // [ H      C.T          G.T     ]
  // [ 0   -r2 * I_y        0      ]
  // [ 0       0       -sigma^{-1} ]
  // Above, sigma = np.diag(z / (s + (gamma_z + 1/p) * z)).
  const int x_dim = input.H.rows;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  const double *s = input.s;
  const double *z = input.z;

  double *sigma = workspace.miscellaneous_workspace.sigma;

  for (int i = 0; i < s_dim; ++i) {
    sigma[i] =
        z[i] /
        (s[i] +
         (input.r3 + (settings.enable_elastics ? 1.0 / input.p : 0.0)) * z[i]);
  }

  SparseMatrix &lhs = workspace.kkt_workspace.lhs;

  lhs.rows = x_dim + y_dim + s_dim;
  lhs.cols = lhs.rows;
  lhs.is_transposed = false;

  int k = 0;

  lhs.indptr[0] = k;

  // Fill upper_hessian_f.
  for (int i = 0; i < x_dim; ++i) {
    for (int j = input.H.indptr[i]; j < input.H.indptr[i + 1]; ++j) {
      if (input.H.ind[j] <= i) {
        lhs.ind[k] = input.H.ind[j];
        lhs.data[k] = input.H.data[j];
        ++k;
      }
    }
    lhs.indptr[i + 1] = k;
  }

  // Fill jacobian_c_t and -gamma_y * I_y.
  for (int i = 0; i < y_dim; ++i) {
    // Fill jacobian_c_t column.
    // NOTE: input.C.is_transposed == true.
    for (int j = input.C.indptr[i]; j < input.C.indptr[i + 1]; ++j) {
      lhs.ind[k] = input.C.ind[j];
      lhs.data[k] = input.C.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = -input.r2;
    ++k;
    lhs.indptr[x_dim + i + 1] = k;
  }

  // Fill jacobian_g_t and -sigma^{-1}.
  for (int i = 0; i < s_dim; ++i) {
    // Fill jacobian_g_t column.
    for (int j = input.G.indptr[i]; j < input.G.indptr[i + 1]; ++j) {
      lhs.ind[k] = input.G.ind[j];
      lhs.data[k] = input.G.data[j];
      ++k;
    }
    // Fill -sigma^{-1} column.
    lhs.ind[k] = x_dim + y_dim + i;
    lhs.data[k] = -1.0 / sigma[i];
    ++k;
    lhs.indptr[x_dim + y_dim + i + 1] = k;
  }
}

void build_nrhs_3x3(const Input &input, const Settings &settings,
                    Workspace &workspace) {
  // Builds the following vector:
  // [ grad_f + C.T @ y + G.T @ z  ]
  // [              c              ]
  // [ g(x) + input.mu / z - z / p ]
  const double *y = input.y;
  const double *z = input.z;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(input.grad_f, input.grad_f + x_dim, nrhs);

  add_ATx_to_y(input.C, y, nrhs);
  add_ATx_to_y(input.G, z, nrhs);

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + i] = input.c[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + y_dim + i] = input.g[i] + input.mu / z[i] -
                              (settings.enable_elastics ? z[i] / input.p : 0.0);
  }
}

void compute_search_direction_3x3(const Input &input, const Settings &settings,
                                  Workspace &workspace, Output &output) {
  build_lhs_3x3(input, settings, workspace);
  build_nrhs_3x3(input, settings, workspace);

  if (settings.permute_kkt_system) {
    assert(settings.kkt_p != nullptr);
    assert(settings.kkt_pinv != nullptr);
    int *AtoC = nullptr;
    permute(workspace.kkt_workspace.lhs, settings.kkt_pinv,
            workspace.miscellaneous_workspace.permutation_workspace, AtoC,
            workspace.kkt_workspace.permuted_lhs);
  }

  const auto &lhs = settings.permute_kkt_system
                        ? workspace.kkt_workspace.permuted_lhs
                        : workspace.kkt_workspace.lhs;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  const int dim = x_dim + s_dim + y_dim;

  [[maybe_unused]] const int sumLnz = QDLDL_etree(
      lhs.rows, lhs.indptr, lhs.ind, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  [[maybe_unused]] const int num_pos_D_entries = QDLDL_factor(
      dim, lhs.indptr, lhs.ind, lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] =
          -workspace.kkt_workspace.negative_rhs[settings.kkt_p[i]];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  add_Ax_to_y_where_A_upper_symmetric(
      lhs, workspace.qdldl_workspace.x,
      workspace.miscellaneous_workspace.lin_sys_residual);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[i]];
    }

    int offset = x_dim;

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[offset + i]];
    }

    offset += y_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.dz[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[offset + i]];
    }
  } else {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[i];
    }

    int offset = x_dim;

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[offset + i];
    }

    offset += y_dim;

    for (int i = 0; i < s_dim; ++i) {
      output.dz[i] = workspace.qdldl_workspace.x[offset + i];
    }
  }

  // ds = -s / z * dz - s + mu / z;
  for (int i = 0; i < s_dim; ++i) {
    output.ds[i] = -input.s[i] / input.z[i] * output.dz[i] - input.s[i] +
                   input.mu / input.z[i];
  }

  // de = -e - (dz + z) / p
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      const double p = input.p;
      output.de[i] = -input.e[i] - (output.dz[i] + input.z[i]) / p;
    }
  }

  output.lin_sys_error = 0.0;

  for (int i = 0; i < dim; ++i) {
    output.lin_sys_error = std::max(
        output.lin_sys_error,
        std::fabs(workspace.miscellaneous_workspace.lin_sys_residual[i]));
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.lin_sys_error = std::max(
          std::fabs((output.dz[i] + output.de[i] + input.z[i]) / input.p +
                    input.e[i]),
          output.lin_sys_error);
    }
  }

  for (int i = 0; i < s_dim; ++i) {
    output.lin_sys_error =
        std::max(std::fabs(output.ds[i] * input.z[i] / input.s[i] +
                           output.dz[i] - input.mu / input.s[i] + input.z[i]),
                 output.lin_sys_error);
  }

  output.kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    output.kkt_error = std::max(
        output.kkt_error, std::fabs(workspace.kkt_workspace.negative_rhs[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    output.kkt_error = std::max(output.kkt_error, std::fabs(input.c[i]));
  }
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error = std::max(
          output.kkt_error, std::fabs(input.g[i] + input.s[i] + input.e[i]));
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error =
          std::max(output.kkt_error, std::fabs(input.g[i] + input.s[i]));
    }
  }
}

void build_lhs_2x2(const Input &input, const Settings &settings,
                   Workspace &workspace) {
  // Builds the following matrix in CSC format:
  // [ H + G.T @ sigma @ G      C.T    ]
  // [          0            -r2 * I_y ]
  // Above, sigma = np.diag(z / (s + (gamma_z + 1/p) * z)).
  const int x_dim = input.H.rows;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  const double *s = input.s;
  const double *z = input.z;

  SparseMatrix &lhs = workspace.kkt_workspace.lhs;

  double *sigma = workspace.miscellaneous_workspace.sigma;

  for (int i = 0; i < s_dim; ++i) {
    sigma[i] =
        z[i] /
        (s[i] +
         (input.r3 + (settings.enable_elastics ? 1.0 / input.p : 0.0)) * z[i]);
  }

  XT_D_X(input.G, sigma, workspace.miscellaneous_workspace.jac_g_t_sigma_jac_g);

  add(input.H, workspace.miscellaneous_workspace.jac_g_t_sigma_jac_g, lhs);

  lhs.rows += y_dim;
  lhs.cols += y_dim;

  // Fill jacobian_c_t and -gamma_y * I_y.
  int k = lhs.indptr[x_dim];
  for (int i = 0; i < y_dim; ++i) {
    // Fill jacobian_c_t column.
    // NOTE: input.C.is_transposed == true.
    for (int j = input.C.indptr[i]; j < input.C.indptr[i + 1]; ++j) {
      lhs.ind[k] = input.C.ind[j];
      lhs.data[k] = input.C.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = -input.r2;
    ++k;
    lhs.indptr[x_dim + i + 1] = k;
  }
}

void build_nrhs_2x2(const Input &input, const Settings &settings,
                    Workspace &workspace) {
  // Builds the following vector:
  // [ grad_f + C.T @ y + G.T @ z + G.T @ sigma @ (g(x) + (mu / z) - z / p)  ]
  // [                                    c                                  ]
  const double *y = input.y;
  const double *z = input.z;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  std::copy(input.grad_f, input.grad_f + x_dim,
            workspace.miscellaneous_workspace.grad_x_lagrangian);

  add_ATx_to_y(input.C, y, workspace.miscellaneous_workspace.grad_x_lagrangian);

  add_ATx_to_y(input.G, z, workspace.miscellaneous_workspace.grad_x_lagrangian);

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(workspace.miscellaneous_workspace.grad_x_lagrangian,
            workspace.miscellaneous_workspace.grad_x_lagrangian + x_dim, nrhs);

  double *sigma = workspace.miscellaneous_workspace.sigma;

  for (int i = 0; i < s_dim; ++i) {
    workspace.miscellaneous_workspace
        .sigma_times_g_plus_mu_over_z_minus_z_over_p[i] =
        sigma[i] * (input.g[i] + input.mu / z[i] -
                    (settings.enable_elastics ? z[i] / input.p : 0.0));
  }

  add_ATx_to_y(input.G,
               workspace.miscellaneous_workspace
                   .sigma_times_g_plus_mu_over_z_minus_z_over_p,
               nrhs);

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + i] = input.c[i];
  }
}

void compute_search_direction_2x2(const Input &input, const Settings &settings,
                                  Workspace &workspace, Output &output) {
  build_lhs_2x2(input, settings, workspace);
  build_nrhs_2x2(input, settings, workspace);

  if (settings.permute_kkt_system) {
    assert(settings.kkt_p != nullptr);
    assert(settings.kkt_pinv != nullptr);
    int *AtoC = nullptr;
    permute(workspace.kkt_workspace.lhs, settings.kkt_pinv,
            workspace.miscellaneous_workspace.permutation_workspace, AtoC,
            workspace.kkt_workspace.permuted_lhs);
  }

  const auto &lhs = settings.permute_kkt_system
                        ? workspace.kkt_workspace.permuted_lhs
                        : workspace.kkt_workspace.lhs;

  const int x_dim = input.H.cols;
  const int s_dim = get_s_dim(input.G);
  const int y_dim = get_y_dim(input.C);

  const int dim = x_dim + y_dim;

  [[maybe_unused]] const int sumLnz = QDLDL_etree(
      lhs.rows, lhs.indptr, lhs.ind, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  [[maybe_unused]] const int num_pos_D_entries = QDLDL_factor(
      dim, lhs.indptr, lhs.ind, lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] =
          -workspace.kkt_workspace.negative_rhs[settings.kkt_p[i]];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
      workspace.miscellaneous_workspace.lin_sys_residual[i] =
          -workspace.qdldl_workspace.x[i];
    }
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  add_Ax_to_y_where_A_upper_symmetric(
      lhs, workspace.qdldl_workspace.x,
      workspace.miscellaneous_workspace.lin_sys_residual);

  if (settings.permute_kkt_system) {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[i]];
    }

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[settings.kkt_pinv[x_dim + i]];
    }
  } else {
    for (int i = 0; i < x_dim; ++i) {
      output.dx[i] = workspace.qdldl_workspace.x[i];
    }

    for (int i = 0; i < y_dim; ++i) {
      output.dy[i] = workspace.qdldl_workspace.x[x_dim + i];
    }
  }

  // dz = sigma @ (g(x) + G @ dx + (mu / z - z / p))
  for (int i = 0; i < s_dim; ++i) {
    output.dz[i] = workspace.miscellaneous_workspace
                       .sigma_times_g_plus_mu_over_z_minus_z_over_p[i];
  }

  add_weighted_Ax_to_y(input.G, workspace.miscellaneous_workspace.sigma,
                       output.dx, output.dz);

  // ds = -s / z * dz - s + mu / z;
  for (int i = 0; i < s_dim; ++i) {
    output.ds[i] = -input.s[i] / input.z[i] * output.dz[i] - input.s[i] +
                   input.mu / input.z[i];
  }

  // de = -e - (dz + z) / p
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      const double p = input.p;
      output.de[i] = -input.e[i] - (output.dz[i] + input.z[i]) / p;
    }
  }

  output.lin_sys_error = 0.0;

  for (int i = 0; i < dim; ++i) {
    output.lin_sys_error = std::max(
        output.lin_sys_error,
        std::fabs(workspace.miscellaneous_workspace.lin_sys_residual[i]));
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.lin_sys_error = std::max(
          std::fabs((output.dz[i] + output.de[i] + input.z[i]) / input.p +
                    input.e[i]),
          output.lin_sys_error);
    }
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      workspace.miscellaneous_workspace.z_residual[i] =
          output.ds[i] - input.r3 * output.dz[i] + output.de[i] / input.p +
          input.g[i] + input.s[i] + input.e[i];
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      workspace.miscellaneous_workspace.z_residual[i] =
          output.ds[i] - input.r3 * output.dz[i] + input.g[i] + input.s[i];
    }
  }

  add_Ax_to_y(input.G, output.dx, workspace.miscellaneous_workspace.z_residual);

  for (int i = 0; i < s_dim; ++i) {
    output.lin_sys_error =
        std::max(std::fabs(output.ds[i] * input.z[i] / input.s[i] +
                           output.dz[i] - input.mu / input.s[i] + input.z[i]),
                 output.lin_sys_error);
    output.lin_sys_error =
        std::max(std::fabs(workspace.miscellaneous_workspace.z_residual[i]),
                 output.lin_sys_error);
  }

  output.kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    output.kkt_error = std::max(
        output.kkt_error,
        std::fabs(workspace.miscellaneous_workspace.grad_x_lagrangian[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    output.kkt_error = std::max(output.kkt_error, std::fabs(input.c[i]));
  }
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error = std::max(
          output.kkt_error, std::fabs(input.g[i] + input.s[i] + input.e[i]));
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      output.kkt_error =
          std::max(output.kkt_error, std::fabs(input.g[i] + input.s[i]));
    }
  }
}

void compute_search_direction(const Input &input, const Settings &settings,
                              Workspace &workspace, Output &output) {
  switch (settings.lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return compute_search_direction_4x4(input, settings, workspace, output);
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3:
    return compute_search_direction_3x3(input, settings, workspace, output);
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return compute_search_direction_2x2(input, settings, workspace, output);
  }
}

auto check_inputs([[maybe_unused]] const Input &input,
                  [[maybe_unused]] const Settings &settings) {
  assert(!settings.enable_elastics || input.p > 0.0);
  assert(input.C.is_transposed);
  switch (settings.lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    assert(input.G.is_transposed);
    break;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3:
    assert(input.G.is_transposed);
    break;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    assert(!input.G.is_transposed);
    break;
  }
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

void MiscellaneousWorkspace::reserve(int x_dim, int s_dim, int kkt_dim,
                                     int upper_jac_g_t_jac_g_nnz) {
  g_plus_s = new double[s_dim];
  lin_sys_residual = new double[kkt_dim];
  z_residual = new double[s_dim];
  grad_x_lagrangian = new double[x_dim];
  sigma = new double[s_dim];
  sigma_times_g_plus_mu_over_z_minus_z_over_p = new double[s_dim];
  jac_g_t_sigma_jac_g.reserve(x_dim, upper_jac_g_t_jac_g_nnz);
  permutation_workspace = new int[kkt_dim];
}

void MiscellaneousWorkspace::free() {
  delete[] g_plus_s;
  delete[] lin_sys_residual;
  delete[] z_residual;
  delete[] grad_x_lagrangian;
  delete[] sigma;
  delete[] sigma_times_g_plus_mu_over_z_minus_z_over_p;
  jac_g_t_sigma_jac_g.free();
  delete[] permutation_workspace;
}

auto MiscellaneousWorkspace::mem_assign(int x_dim, int s_dim, int kkt_dim,
                                        int jac_g_t_jac_g_nnz,
                                        unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  g_plus_s = reinterpret_cast<decltype(g_plus_s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  lin_sys_residual =
      reinterpret_cast<decltype(lin_sys_residual)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  z_residual = reinterpret_cast<decltype(z_residual)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  grad_x_lagrangian =
      reinterpret_cast<decltype(grad_x_lagrangian)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  sigma = reinterpret_cast<decltype(sigma)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  sigma_times_g_plus_mu_over_z_minus_z_over_p =
      reinterpret_cast<decltype(sigma_times_g_plus_mu_over_z_minus_z_over_p)>(
          mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  cum_size += jac_g_t_sigma_jac_g.mem_assign(x_dim, jac_g_t_jac_g_nnz,
                                             mem_ptr + cum_size);

  permutation_workspace =
      reinterpret_cast<decltype(permutation_workspace)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  return cum_size;
}

void KKTWorkspace::reserve(int kkt_dim, int kkt_nnz) {
  lhs.reserve(kkt_dim, kkt_nnz);
  permuted_lhs.reserve(kkt_dim, kkt_nnz);
  negative_rhs = new double[kkt_dim];
}

void KKTWorkspace::free() {
  lhs.free();
  permuted_lhs.free();
  delete[] negative_rhs;
}

auto KKTWorkspace::mem_assign(int kkt_dim, int kkt_nnz, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  cum_size += lhs.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);
  cum_size += permuted_lhs.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);

  negative_rhs = reinterpret_cast<decltype(negative_rhs)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  return cum_size;
}

auto get_kkt_dim(Settings::LinearSystemFormulation lin_sys_formulation,
                 int x_dim, int s_dim, int y_dim) -> int {
  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return x_dim + 2 * s_dim + y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3:
    return x_dim + s_dim + y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return x_dim + y_dim;
  }
};

auto get_kkt_nnz(Settings::LinearSystemFormulation lin_sys_formulation,
                 int upper_hessian_f_nnz, int jacobian_c_nnz,
                 int jacobian_g_nnz,
                 int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int s_dim,
                 int y_dim) {
  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return upper_hessian_f_nnz + jacobian_c_nnz + jacobian_g_nnz + 3 * s_dim +
           y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3:
    return upper_hessian_f_nnz + jacobian_c_nnz + jacobian_g_nnz + s_dim +
           y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz + jacobian_c_nnz +
           y_dim;
  }
};

void Workspace::reserve(Settings::LinearSystemFormulation lin_sys_formulation,
                        int x_dim, int s_dim, int y_dim,
                        int upper_hessian_f_nnz, int jacobian_c_nnz,
                        int jacobian_g_nnz, int upper_jac_g_t_jac_g_nnz,
                        int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz,
                        int kkt_L_nnz) {
  const int kkt_dim = get_kkt_dim(lin_sys_formulation, x_dim, s_dim, y_dim);
  const int kkt_nnz = get_kkt_nnz(
      lin_sys_formulation, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, s_dim, y_dim);

  kkt_workspace.reserve(kkt_dim, kkt_nnz);
  qdldl_workspace.reserve(kkt_dim, kkt_L_nnz);
  miscellaneous_workspace.reserve(x_dim, s_dim, kkt_dim,
                                  upper_jac_g_t_jac_g_nnz);
}

void Workspace::free() {
  kkt_workspace.free();
  qdldl_workspace.free();
  miscellaneous_workspace.free();
}

auto Workspace::mem_assign(
    Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim,
    int y_dim, int upper_hessian_f_nnz, int jacobian_c_nnz,
    int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
    int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int kkt_L_nnz,
    unsigned char *mem_ptr) -> int {
  const int kkt_dim = get_kkt_dim(lin_sys_formulation, x_dim, s_dim, y_dim);
  const int kkt_nnz = get_kkt_nnz(
      lin_sys_formulation, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, s_dim, y_dim);

  int cum_size = 0;

  cum_size += kkt_workspace.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);
  cum_size +=
      qdldl_workspace.mem_assign(kkt_dim, kkt_L_nnz, mem_ptr + cum_size);
  cum_size += miscellaneous_workspace.mem_assign(
      x_dim, s_dim, kkt_dim, jac_g_t_jac_g_nnz, mem_ptr + cum_size);

  return cum_size;
}

} // namespace sip_qdldl
