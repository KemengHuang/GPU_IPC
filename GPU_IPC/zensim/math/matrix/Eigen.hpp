#pragma once

#include "Utility.h"
#include "../Vec.h"

namespace zs {

  /// ref: http://docs.ros.org/en/kinetic/api/gtsam/html/SelfAdjointEigenSolver_8h_source.html
  // This file is part of Eigen, a lightweight C++ template library
  // for linear algebra.
  //
  // Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
  // Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
  //
  // This Source Code Form is subject to the terms of the Mozilla
  // Public License v. 2.0. If a copy of the MPL was not distributed
  // with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
  template <typename VecTM,
            enable_if_all<VecTM::dim == 2,
                          VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                          VecTM::template range_t<0>::value == 2> = 0>
  constexpr auto eigen_decomposition(const VecInterface<VecTM> &M) noexcept {
    using value_type = typename VecTM::value_type;
    using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                            conditional_t<(sizeof(value_type) >= 8), f64, f32>>;
    using MatT = typename VecTM::template variant_vec<T, typename VecTM::extents>;
    using VecT =
        typename VecTM::template variant_vec<T, integer_seq<typename VecTM::index_type, 2>>;
    MatT eivecs{};
    VecT eivals{};
    T shift = trace(M) / (T)2;
    auto scaledMat = M.clone();
    scaledMat(0, 1) = M(1, 0);
    scaledMat(0, 0) -= shift;
    scaledMat(1, 1) -= shift;
    T scale = scaledMat.abs().max();
    if (scale > (T)0) scaledMat /= scale;
    // compute eigenvalues
    {
      T t0 = (T)0.5
             * zs::sqrt(zs::sqr(scaledMat(0, 0) - scaledMat(1, 1))
                        + (T)4 * zs::sqr(scaledMat(1, 0)));
      T t1 = (T)0.5 * (scaledMat(0, 0) + scaledMat(1, 1));
      eivals(0) = t1 - t0;
      eivals(1) = t1 + t0;
    }
    // compute eigenvectors
    {
      if (eivals(1) - eivals(0) <= zs::abs(eivals(1)) * limits<T>::epsilon())
        eivecs = MatT::identity();
      else {
        for (int d = 0; d != 2; ++d) scaledMat(d, d) -= eivals(1);
        const T a2 = zs::sqr(scaledMat(0, 0));
        const T c2 = zs::sqr(scaledMat(1, 1));
        const T b2 = zs::sqr(scaledMat(1, 0));
        if (a2 > c2) {
          auto coeff = zs::sqrt(a2 + b2);
          eivecs(0, 1) = -scaledMat(1, 0) / coeff;
          eivecs(1, 1) = scaledMat(0, 0) / coeff;
        } else {
          auto coeff = zs::sqrt(c2 + b2);
          eivecs(0, 1) = -scaledMat(1, 1) / coeff;
          eivecs(1, 1) = scaledMat(1, 0) / coeff;
        }
        auto c0 = col(eivecs, 1).orthogonal();
        eivecs(0, 0) = c0(0);
        eivecs(1, 0) = c0(1);
      }
    }
    eivals *= scale;
    eivals += shift;
    return zs::make_tuple(eivals, eivecs);
  }

  template <typename VecTM,
            enable_if_all<VecTM::dim == 2,
                          VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                          VecTM::template range_t<0>::value == 3> = 0>
  constexpr auto eigen_decomposition(const VecInterface<VecTM> &mat) noexcept {
    using value_type = typename VecTM::value_type;
    using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                            conditional_t<(sizeof(value_type) >= 8), f64, f32>>;
    using MatT = typename VecTM::template variant_vec<T, typename VecTM::extents>;
    using VecT =
        typename VecTM::template variant_vec<T, integer_seq<typename VecTM::index_type, 3>>;
    MatT eivecs{};
    VecT eivals{};
    // Shift the matrix to the mean eigenvalue and map the matrix coefficients to [-1:1] to avoid
    // over- and underflow.
    T shift = trace(mat) / (T)3;
    auto scaledMat = mat.clone();
    scaledMat(0, 0) -= shift;
    scaledMat(1, 1) -= shift;
    scaledMat(2, 2) -= shift;
    T scale = scaledMat.abs().max();
    if (scale > (T)0) scaledMat /= scale;
    // compute eigenvalues
    {
      const T s_inv3 = (T)1 / (T)3;
      const T s_sqrt3 = zs::sqrt((T)3);
      // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
      // eigenvalues are the roots to this equation, all guaranteed to be
      // real-valued, because the matrix is symmetric.
      T c0 = scaledMat(0, 0) * scaledMat(1, 1) * scaledMat(2, 2)
             + (T)2 * scaledMat(1, 0) * scaledMat(2, 0) * scaledMat(2, 1)
             - scaledMat(0, 0) * scaledMat(2, 1) * scaledMat(2, 1)
             - scaledMat(1, 1) * scaledMat(2, 0) * scaledMat(2, 0)
             - scaledMat(2, 2) * scaledMat(1, 0) * scaledMat(1, 0);
      T c1 = scaledMat(0, 0) * scaledMat(1, 1) - scaledMat(1, 0) * scaledMat(1, 0)
             + scaledMat(0, 0) * scaledMat(2, 2) - scaledMat(2, 0) * scaledMat(2, 0)
             + scaledMat(1, 1) * scaledMat(2, 2) - scaledMat(2, 1) * scaledMat(2, 1);
      T c2 = scaledMat(0, 0) + scaledMat(1, 1) + scaledMat(2, 2);

      // Construct the parameters used in classifying the roots of the equation
      // and in solving the equation for the roots in closed form.
      T c2_over_3 = c2 * s_inv3;
      T a_over_3 = (c2 * c2_over_3 - c1) * s_inv3;
      a_over_3 = zs::max(a_over_3, (T)0);

      T half_b = (T)(0.5) * (c0 + c2_over_3 * ((T)(2) * c2_over_3 * c2_over_3 - c1));

      T q = a_over_3 * a_over_3 * a_over_3 - half_b * half_b;
      q = zs::max(q, (T)0);

      // Compute the eigenvalues by solving for the roots of the polynomial.
      T rho = zs::sqrt(a_over_3);
      // since sqrt(q) > 0, atan2 is in [0, pi] and theta is in [0, pi/3]
      T theta = zs::atan2(zs::sqrt(q), half_b) * s_inv3;
      T cos_theta = zs::cos(theta);
      T sin_theta = zs::sin(theta);
      // roots are already sorted, since cos is monotonically decreasing on [0, pi]
      eivals(0) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);  // == 2*rho*cos(theta+2pi/3)
      eivals(1) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);  // == 2*rho*cos(theta+ pi/3)
      eivals(2) = c2_over_3 + (T)2 * rho * cos_theta;
    }
    // compute eigenvectors
    {
      if ((eivals(2) - eivals(0)) <= limits<T>::epsilon()) {
        // All three eigenvalues are numerically the same
        eivecs = MatT::identity();
      } else {
        auto tmp = scaledMat.clone();

        // Compute the eigenvector of the most distinct eigenvalue
        T d0 = eivals(2) - eivals(1);
        T d1 = eivals(1) - eivals(0);
        int k{0}, l{2};
        if (d0 > d1) {
          // std::swap(k, l); // damn you nvcc!
          auto t = k;
          k = l;
          l = t;
          d0 = d1;
        }

        auto extractKernel = [&eivecs](MatT &mat, int colNo) -> VecT {
          VecT representative{};
          int i0{};
          // find non-zero column i0 (there must exist a non zero coeff on diagonal)
          T entry{limits<T>::lowest()};
          for (int d = 0; d != 3; ++d)
            if (auto v = zs::abs(mat(d, d)); v > entry) {
              entry = v;
              i0 = d;
            }
          // mat.col(i0) is a good candidate for an orthogonal vector to the current eigenvector
          representative = col(mat, i0);
          T n0{}, n1{};
          VecT c0{}, c1{};
          n0 = (c0 = representative.cross(col(mat, (i0 + 1) % 3))).l2NormSqr();
          n1 = (c1 = representative.cross(col(mat, (i0 + 2) % 3))).l2NormSqr();
          if (n0 > n1) {
            auto coeff = zs::sqrt(n0);
            for (int d = 0; d != 3; ++d) eivecs(d, colNo) = c0(d) / coeff;
          } else {
            auto coeff = zs::sqrt(n1);
            for (int d = 0; d != 3; ++d) eivecs(d, colNo) = c1(d) / coeff;
          }
          return representative;
        };
        // Compute the eigenvector of index k
        {
          tmp(0, 0) -= eivals(k);
          tmp(1, 1) -= eivals(k);
          tmp(2, 2) -= eivals(k);
          // 'tmp' is of rank 2, and its kernel corresponds to the respective eigenvector
          auto colL = extractKernel(tmp, k);
          for (int d = 0; d != 3; ++d) eivecs(d, l) = colL(d);
        }

        // Compute eigenvector of index l
        if (d0 <= 2 * limits<T>::epsilon() * d1) {
          // If d0 is too small, then the two other eigenvalues are numerically the same,
          // and thus we only have to ortho-normalize the near orthogonal vector we saved above.
          auto colL = col(eivecs, l);
          auto t = (colL - col(eivecs, k).dot(colL) * colL).normalized();
          for (int d = 0; d != 3; ++d) eivecs(d, l) = t(d);
        } else {
          tmp = scaledMat;
          tmp(0, 0) -= eivals(l);
          tmp(1, 1) -= eivals(l);
          tmp(2, 2) -= eivals(l);

          extractKernel(tmp, l);
        }

        // Compute last eigenvector from the other two
        auto c1 = col(eivecs, 2).cross(col(eivecs, 0)).normalized();
        for (int d = 0; d != 3; ++d) eivecs(d, 1) = c1(d);
      }
    }
    eivals *= scale;
    eivals += shift;
    return zs::make_tuple(eivals, eivecs);
  }

  /// ref: Yu Fang, wiki
  template <
      typename VecTM,
      enable_if_all<
          VecTM::dim == 2, VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
          VecTM::template range_t<0>::value != 2, VecTM::template range_t<0>::value != 3> = 0>
  constexpr auto eigen_decomposition(const VecInterface<VecTM> &mat) noexcept {
    using value_type = typename VecTM::value_type;
    using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                            conditional_t<(sizeof(value_type) >= 8), f64, f32>>;
    using Ti = typename VecTM::index_type;
    constexpr int dim = VecTM::template range_t<0>::value;
    using MatT = typename VecTM::template variant_vec<T, typename VecTM::extents>;
    using VecT =
        typename VecTM::template variant_vec<T, integer_seq<typename VecTM::index_type, dim>>;
    using IVecT =
        typename VecTM::template variant_vec<Ti, integer_seq<typename VecTM::index_type, dim>>;

    auto S = mat.clone();
    MatT E = MatT::identity();  // eigen vectors
    VecT e{};                   // eigen values
    T max_S = 0;
    for (Ti i = 0; i != dim; ++i)
      for (Ti j = 0; j != dim; ++j) max_S = zs::max(max_S, zs::abs(S(i, j)));
    Ti k{}, l{}, m{}, state{};
    T s{}, c{}, t{}, p{}, y{}, d{}, r{};
    IVecT ind{}, changed{};
    state = dim;
    auto maxind = [&S](Ti k) {
      constexpr int dim = VecTM::template range_t<0>::value;
      Ti m = k + 1;
      for (Ti i = k + 2; i < dim; ++i)
        if (zs::abs(S(k, i)) > zs::abs(S(k, m))) m = i;
      return m;
    };
    auto update = [&e, &changed, &y, &state](int k, T t) {
      y = e[k];
      e[k] = y + t;
      if (changed[k] && y == e[k]) {
        changed[k] = false;
        --state;
      } else if (!changed[k] && y != e[k]) {
        changed[k] = true;
        ++state;
      }
    };
    auto rotate = [&S](Ti k, Ti l, Ti i, Ti j, T s, T c) {
      constexpr int dim = VecTM::template range_t<0>::value;
      T Skl = S(k, l), Sij = S(i, j);
      S(k, l) = c * Skl - s * Sij;
      S(i, j) = s * Skl + c * Sij;
    };
    for (k = 0; k != dim; ++k) {
      ind[k] = maxind(k);
      e[k] = S(k, k);
      changed[k] = true;
    }
    while (state != 0) {
      m = 0;
      for (k = 1; k < dim - 1; ++k)
        if (zs::abs(S(k, ind[k])) > zs::abs(S(m, ind[m]))) m = k;
      k = m;
      l = ind[m];
      p = S(k, l);
      if (zs::abs(p) < abs(max_S) * 1e-6) break;
      y = (e[l] - e[k]) / 2;
      d = zs::abs(y) + zs::sqrt(p * p + y * y);
      r = zs::sqrt(p * p + d * d);
      c = d / r;
      s = p / r;
      t = p * p / d;
      if (y < 0) {
        s = -s;
        t = -t;
      }
      S(k, l) = 0;
      update(k, -t);
      update(l, t);
      for (int i = 0; i <= k - 1; ++i) rotate(i, k, i, l, s, c);
      for (int i = k + 1; i <= l - 1; ++i) rotate(k, i, i, l, s, c);
      for (int i = l + 1; i < dim; ++i) rotate(k, i, l, i, s, c);
      for (int i = 0; i != dim; ++i) {
        T Eik = E(i, k), Eil = E(i, l);
        E(i, k) = c * Eik - s * Eil;
        E(i, l) = s * Eik + c * Eil;
      }
      ind[k] = maxind(k);
      ind[l] = maxind(l);
    }
    return zs::make_tuple(e, E);
  }

  template <
      typename VecTM,
      enable_if_all<std::is_floating_point_v<typename VecTM::value_type>, VecTM::dim == 2,
                    VecTM::template range_t<0>::value == VecTM::template range_t<1>::value> = 0>
  constexpr void make_pd(VecInterface<VecTM> &mat) noexcept {
    constexpr int dim = VecTM::template range_t<0>::value;
    using value_type = typename VecTM::value_type;
    using MatT = typename VecTM::template variant_vec<value_type, typename VecTM::extents>;

    // ref:
    // Hierarchical Optimization Time Integration (HOT)
    // https://github.com/penn-graphics-research/HOT/blob/d8d57be410ed343c3fb37af6020cf5e14a0d1bec/Lib/Ziran/Math/Linear/EigenDecomposition.h#L111
    auto [eivals, eivecs] = eigen_decomposition(mat);
    for (int i = 0; i != dim; ++i) {
      if (eivals[i] < 0)
        eivals[i] = 0;
      //else
      //  break;  // eivals in ascending order
    }
#if 0
    auto diag = MatT::zeros();
    for (int d = 0; d != dim; ++d) diag(d, d) = eivals[d];
    mat.assign(eivecs * diag * eivecs.transpose());
#else
    mat.assign(mat.zeros());
    for (int i = 0; i != dim; ++i)
      for (int j = 0; j != dim; ++j)
        for (int k = 0; k != dim; ++k) mat(i, k) += eivecs(i, j) * eivals[j] * eivecs(k, j);
#endif
  }

}  // namespace zs