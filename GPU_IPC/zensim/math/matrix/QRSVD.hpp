#pragma once
#include "Givens.hpp"
#include "../Vec.h"

/// <summary>
/// qrsvd
///  ref:
///  https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Math/Linear/ImplicitQRSVD.h
/// </summary>

namespace zs {

  namespace math {

    template <typename VecT, enable_if_all<vec_fits_shape<VecT, 2, 2, 2>(),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto polar_decomposition(const VecInterface<VecT>& A,
                                       GivensRotation<typename VecT::value_type>& R) noexcept {
      using value_type = typename VecT::value_type;
      // constexpr auto N = VecT::template range_t<0>::value;
      typename VecT::template variant_vec<value_type, typename VecT::extents> S = A;
      vec<value_type, 2> x{A(0, 0) + A(1, 1), A(1, 0) - A(0, 1)};
      auto d = x.norm();
      if (d != 0) {
        R.c = x(0) / d;
        R.s = -x(1) / d;
      } else {
        R.c = 1;
        R.s = 0;
      }
      R.rowRotation(S);
      return S;
    }

    template <typename VecT,
              enable_if_all<VecT::dim == 2,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            VecT::template range_t<0>::value == 2,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto qr_svd(const VecInterface<VecT>& A, GivensRotation<typename VecT::value_type>& U,
                          GivensRotation<typename VecT::value_type>& V) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range_t<0>::value;
      typename VecT::template variant_vec<value_type, integer_seq<index_type, N>> S{};

      auto S_sym = polar_decomposition(A, U);
      value_type cosine{}, sine{};
      auto x{S_sym(0, 0)}, y{S_sym(0, 1)}, z{S_sym(1, 1)};
      auto y2 = y * y;

      if (y2 == 0) {  // S is already diagonal
        cosine = 1;
        sine = 0;
        S(0) = x;
        S(1) = z;
      } else {
        auto tau = (value_type)0.5 * (x - z);
        value_type w{zs::sqrt(tau * tau + y2)}, t{};
        if (tau > 0)  // tau + w > w > y > 0 ==> division is safe
          t = y / (tau + w);
        else  // tau - w < -w < -y < 0 ==> division is safe
          t = y / (tau - w);
        cosine = zs::rsqrt(t * t + (value_type)1);
        sine = -t * cosine;
        /*
          V = [cosine -sine; sine cosine]
          Sigma = V'SV. Only compute the diagonals for efficiency.
          Also utilize symmetry of S and don't form V yet.
        */
        value_type c2 = cosine * cosine;
        value_type _2csy = 2 * cosine * sine * y;
        value_type s2 = sine * sine;

        S(0) = c2 * x - _2csy + s2 * z;
        S(1) = s2 * x + _2csy + c2 * z;
      }
      // Sorting
      // Polar already guarantees negative sign is on the small magnitude singular value.
      if (S(0) < S(1)) {
        // std::swap(S(0), S(1));
        auto S0 = S(0);
        S(0) = S(1);
        S(1) = S0;
        V.c = -sine;
        V.s = cosine;
      } else {
        V.c = cosine;
        V.s = sine;
      }
      U *= V;
      return S;
    }

    namespace detail {
      /**
       *    \brief  compute wilkinsonShift of the block
       *        a1     b1
       *        b1     a2
       *    based on the wilkinsonShift formula
       *    mu = a2 + d - sign(d) \sqrt (d * d + b1 * b1), where d = (a1-a2)/2
       *    \note   This shift gives average cubic convergence rate for reducing Tn,n−1 to zero
       */
      template <typename T>
      constexpr T wilkinson_shift(const T a1, const T b1, const T a2) noexcept {
        T d = (T)0.5 * (a1 - a2);
        T bs = b1 * b1;

        T mu = a2 - ::zs::copysign(bs / (::zs::abs(d) + ::zs::sqrt(d * d + bs)), d);
        // T mu = a2 - bs / ( d + sign_d*sqrt (d*d + bs));
        return mu;
      }

      template <typename VecTM, typename VecTV,
                enable_if_all<vec_fits_shape<VecTM, 2, 3, 3>(), vec_fits_shape<VecTV, 1, 3>()> = 0>
      constexpr void flip_sign(int j, VecInterface<VecTM>& U, VecInterface<VecTV>& S) noexcept {
        S(j) = -S(j);
        U(0, j) = -U(0, j);
        U(1, j) = -U(1, j);
        U(2, j) = -U(2, j);
      }
      template <int t, typename VecTU, typename VecTS, typename VecTV,
                enable_if_all<vec_fits_shape<VecTU, 2, 3, 3>(), vec_fits_shape<VecTS, 1, 3>(),
                              vec_fits_shape<VecTV, 2, 3, 3>()> = 0>
      constexpr void sort_sigma(VecInterface<VecTU>& U, VecInterface<VecTS>& sigma,
                                VecInterface<VecTV>& V) noexcept {
        const auto swapScalar = [](auto& a, auto& b) {
          auto tmp = a;
          a = b;
          b = tmp;
        };
        /// t == 0
        if constexpr (t == 0) {
          // Case: sigma(0) > |sigma(1)| >= |sigma(2)|
          if (abs(sigma(1)) >= abs(sigma(2))) {
            if (sigma(1) < 0) {
              flip_sign(1, U, sigma);
              flip_sign(2, U, sigma);
            }
            return;
          }

          // fix sign of sigma for both cases
          if (sigma(2) < 0) {
            flip_sign(1, U, sigma);
            flip_sign(2, U, sigma);
          }

          // swap sigma(1) and sigma(2) for both cases
          swapScalar(sigma(1), sigma(2));
          for (int d = 0; d != 3; ++d) {
            swapScalar(U(d, 1), U(d, 2));
            swapScalar(V(d, 1), V(d, 2));
          }

          // Case: |sigma(2)| >= sigma(0) > |simga(1)|
          if (sigma(1) > sigma(0)) {
            swapScalar(sigma(0), sigma(1));
            for (int d = 0; d != 3; ++d) {
              swapScalar(U(d, 0), U(d, 1));
              swapScalar(V(d, 0), V(d, 1));
            }
          }

          // Case: sigma(0) >= |sigma(2)| > |simga(1)|
          else {
            for (int d = 0; d != 3; ++d) {
              U(d, 2) = -U(d, 2);
              V(d, 2) = -V(d, 2);
            }
          }
        }
        /// t == 1
        else if constexpr (t == 1) {
          // Case: |sigma(0)| >= sigma(1) > |sigma(2)|
          if (abs(sigma(0)) >= sigma(1)) {
            if (sigma(0) < 0) {
              flip_sign(0, U, sigma);
              flip_sign(2, U, sigma);
            }
            return;
          }

          // swap sigma(0) and sigma(1) for both cases
          swapScalar(sigma(0), sigma(1));
          // U.col(0).swap(U.col(1));
          // V.col(0).swap(V.col(1));
          for (int d = 0; d != 3; ++d) {
            swapScalar(U(d, 0), U(d, 1));
            swapScalar(V(d, 0), V(d, 1));
          }

          // Case: sigma(1) > |sigma(2)| >= |sigma(0)|
          if (fabs(sigma(1)) < fabs(sigma(2))) {
            swapScalar(sigma(1), sigma(2));
            // U.col(1).swap(U.col(2));
            // V.col(1).swap(V.col(2));
            for (int d = 0; d != 3; ++d) {
              swapScalar(U(d, 1), U(d, 2));
              swapScalar(V(d, 1), V(d, 2));
            }
          }

          // Case: sigma(1) >= |sigma(0)| > |sigma(2)|
          else {
            // U.col(1) = -U.col(1);
            // V.col(1) = -V.col(1);
            for (int d = 0; d != 3; ++d) {
              U(d, 1) = -U(d, 1);
              V(d, 1) = -V(d, 1);
            }
          }

          // fix sign for both cases
          if (sigma(1) < 0) {
            flip_sign(1, U, sigma);
            flip_sign(2, U, sigma);
          }
        }
      }

      template <int t, typename VecTB, typename VecTU, typename VecTS, typename VecTV,
                enable_if_all<vec_fits_shape<VecTB, 2, 3, 3>(), vec_fits_shape<VecTU, 2, 3, 3>(),
                              vec_fits_shape<VecTS, 1, 3>(), vec_fits_shape<VecTV, 2, 3, 3>()> = 0>
      constexpr void process(VecInterface<VecTB>& B, VecInterface<VecTU>& U, VecInterface<VecTS>& S,
                             VecInterface<VecTV>& V) noexcept {
        static_assert(t == 0 || t == 1, "offset t here is not a valid one (0 or 1).");
        using T = typename VecTB::value_type;
        GivensRotation<T> u{0, 1};
        GivensRotation<T> v{0, 1};
        constexpr int other = (t == 1) ? 0 : 2;
        S(other) = B(other, other);
// this is slow (redundant copy) for the moment due to the lack of block view
#if 1
        using mat2 =
            typename VecTB::template variant_vec<T, integer_seq<typename VecTB::index_type, 2, 2>>;
        // using vec2 =
        //     typename VecTB::template variant_vec<T, integer_seq<typename VecTB::index_type, 2>>;
        mat2 B_{};
        B_(0, 0) = B(t, t);
        B_(0, 1) = B(t, t + 1);
        B_(1, 0) = B(t + 1, t);
        B_(1, 1) = B(t + 1, t + 1);
        auto S_ = qr_svd(B_, u, v);  // 2d qrsvd
        B(t, t) = B_(0, 0);
        B(t, t + 1) = B_(0, 1);
        B(t + 1, t) = B_(1, 0);
        B(t + 1, t + 1) = B_(1, 1);
        S(t) = S_(0);
        S(t + 1) = S_(1);
#else
        qr_svd(B.template block<2, 2>(t, t), u, S.template block<2, 1>(t, 0), v);
#endif
        u.rowi += t;
        u.rowk += t;
        v.rowi += t;
        v.rowk += t;
        u.columnRotation(U);
        v.columnRotation(V);
      }
    }  // namespace detail

    // QR decomposition
    template <typename VecT,
              enable_if_all<VecT::dim == 2,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto qr(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range_t<0>::value;
#if 1
      auto Q = VecT::identity();
      auto R = A.clone();

      for (index_type j = 0; j < N; ++j)
        for (index_type i = N - 1; i > j; --i) {
          GivensRotation<value_type> gq{R(i - 1, j), R(i, j), i - 1, i};
          gq.rowRotation(R);
          gq.rowRotation(Q);
        }
      Q = Q.transpose();
      return std::make_tuple(Q, R);
#else
      if constexpr (N == 1) {
        typename VecT::template variant_vec<value_type, typename VecT::extents> Q{}, R{};
        const auto a = A(0, 0);
        R(0, 0) = zs::abs(a);
        Q(0, 0) = a > 0 ? 1 : -1;
        return std::make_tuple(Q, R);
      } else if constexpr (N == 2) {
        GivensRotation<value_type> gq{0, 1};
        gq.computeConventional(A(0, 0), A(1, 0));
        auto Q = VecT::identity();
        auto R = A.clone();
        if (!math::near_zero(R(1, 0))) gq.rowRotation(R);
        gq.fill(Q);
        //
        auto flip_sign = [&Q, &R](int j) {
          if (const auto rjj = R(j, j); rjj < 0) {
            R(j, j) = -rjj;
            Q(0, j) = -Q(0, j);
            Q(1, j) = -Q(1, j);
          }
        };
        flip_sign(0);
        flip_sign(1);
        return std::make_tuple(Q, R);
      } else {
        auto Q = VecT::identity();
        auto R = A.clone();
        auto zero_chase = [&Q, &R](GivensRotation<value_type> gq) {
          gq.rowRotation(R);
          auto G = VecT::identity();
          gq.fill(G);
          Q = Q * G;
        };
        for (int j = 0; j != N; ++j)
          for (int i = N - 1; i != j; --i) {
            if (const auto entry = R(i, j); !math::near_zero(entry))
              zero_chase({R(i - 1, j), entry, i - 1, i});
          }
        //
        auto flip_sign = [&Q, &R](int j) {
          if (const auto rjj = R(j, j); rjj < 0) {
            R(j, j) = -rjj;
            for (int i = 0; i != N; ++i) Q(i, j) = -Q(i, j);
          }
        };
        for (int j = 0; j != N; ++j) flip_sign(j);
        return std::make_tuple(Q, R);
      }
#endif
    }

    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto gram_schmidt(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range_t<0>::value;
      using RetT = typename VecT::template variant_vec<value_type, typename VecT::extents>;
      auto d1 = col(A, 0);
      auto d2 = col(A, 1);
      auto d3 = col(A, 2);
      auto q1 = d1.normalized();
      auto r11 = d1.norm();
      auto r12 = d2.dot(q1);
      auto q2 = d2 - r12 * q1;
      auto r22 = q2.norm();
      q2 = q2.normalized();

      auto r13 = d3.dot(q1);
      auto r23 = d3.dot(q2);
      auto q3 = d3 - r13 * q1 - r23 * q2;
      auto r33 = q3.norm();
      q3 = q3.normalized();

      RetT Q{};
      for (int d = 0; d != 3; ++d) {
        Q(d, 0) = q1(d);
        Q(d, 1) = q2(d);
        Q(d, 2) = q3(d);
      }
      auto R = RetT::zeros();
      R(0, 0) = r11;
      R(0, 1) = r12;
      R(0, 2) = r13;
      R(1, 1) = r22;
      R(1, 2) = r23;
      R(2, 2) = r33;
      return zs::make_tuple(Q, R);
    }

    // Polar guarantees negative sign is on the small magnitude singular value.
    // S is guaranteed to be the closest one to identity.
    // R is guaranteed to be the closest rotation to A.
    template <typename VecT,
              enable_if_all<VecT::dim == 2,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto qr_svd(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range_t<0>::value;
      typename VecT::template variant_vec<value_type, typename VecT::extents> U{}, V{};
      if constexpr (N == 1) {
        typename VecT::template variant_vec<value_type, integer_seq<index_type, N>> S{A(0, 0)};
        U(0, 0) = V(0, 0) = 1;
        return std::make_tuple(U, S, V);
      } else if constexpr (N == 2) {
        GivensRotation<value_type> gu{0, 1}, gv{0, 1};
        auto S = qr_svd(A, gu, gv);
        gu.fill(U);
        gv.fill(V);
        return std::make_tuple(U, S, V);
      } else if constexpr (N == 3) {
        typename VecT::template variant_vec<value_type, integer_seq<index_type, N>> S{};
        auto B = A.clone();
        U = U.identity();
        V = V.identity();
        upper_bidiagonalize(B, U, V);

        GivensRotation<value_type> r{0, 1};

        value_type alpha[3] = {B(0, 0), B(1, 1), B(2, 2)};
        value_type beta[2] = {B(0, 1), B(1, 2)};
        // offdiagonal entries in tridiagonal B^T B
        value_type gamma[2] = {alpha[0] * beta[0], alpha[1] * beta[1]};

        // terminate once any of(α1 β1 α2 β2 α3) becomes smaller than a tolerance τ
        constexpr auto eta = limits<value_type>::epsilon() * (value_type)128;
        value_type tol
            = eta
              * math::max((value_type)0.5
                              * sqrt(alpha[0] * alpha[0] + alpha[1] * alpha[1] + alpha[2] * alpha[2]
                                     + beta[0] * beta[0] + beta[1] * beta[1]),
                          (value_type)1);

        // do implicit shift QR until A^T A is block diagonal
        while (abs(alpha[0]) > tol && abs(alpha[1]) > tol && abs(alpha[2]) > tol
               && abs(beta[0]) > tol && abs(beta[1]) > tol) {
          auto mu = detail::wilkinson_shift(alpha[1] * alpha[1] + beta[0] * beta[0], gamma[1],
                                            alpha[2] * alpha[2] + beta[1] * beta[1]);
          r.computeConventional(alpha[0] * alpha[0] - mu, gamma[0]);
          r.columnRotation(B);

          r.columnRotation(V);
          zero_chasing(B, U, V);

          alpha[0] = B(0, 0);
          alpha[1] = B(1, 1);
          alpha[2] = B(2, 2);
          beta[0] = B(0, 1);
          beta[1] = B(1, 2);
          gamma[0] = alpha[0] * beta[0];
          gamma[1] = alpha[1] * beta[1];
        }

        /**
         *  Handle the cases of one of the alphas and betas being 0
         *  Sorted by ease of handling and then frequency
         *  of occurrence
         *  If B is of form
         *      x x 0
         *      0 x 0
         *      0 0 x
         **/
        if (abs(beta[1]) <= tol) {
          detail::process<0>(B, U, S, V);
          detail::sort_sigma<0>(U, S, V);
        }
        /**
         *  If B is of form
         *      x 0 0
         *      0 x x
         *      0 0 x
         **/
        else if (abs(beta[0]) <= tol) {
          detail::process<1>(B, U, S, V);
          detail::sort_sigma<1>(U, S, V);
        }
        /**
         *  If B is of form
         *      x x 0
         *      0 0 x
         *      0 0 x
         **/
        else if (abs(alpha[1]) <= tol) {
          /**
           *    Reduce B to
           *        x x 0
           *        0 0 0
           *        0 0 x
           **/
          GivensRotation<value_type> r1(1, 2);
          r1.computeUnconventional(B(1, 2), B(2, 2));
          r1.rowRotation(B);
          r1.columnRotation(U);

          detail::process<0>(B, U, S, V);
          detail::sort_sigma<0>(U, S, V);
        }
        /**
         *  If B is of form
         *      x x 0
         *      0 x x
         *      0 0 0
         **/
        else if (abs(alpha[2]) <= tol) {
          /**
           *    Reduce B to
           *        x x +
           *        0 x 0
           *        0 0 0
           **/
          GivensRotation<value_type> r1(1, 2);
          r1.computeConventional(B(1, 1), B(1, 2));
          r1.columnRotation(B);
          r1.columnRotation(V);
          /**
           *    Reduce B to
           *        x x 0
           *        + x 0
           *        0 0 0
           **/
          GivensRotation<value_type> r2(0, 2);
          r2.computeConventional(B(0, 0), B(0, 2));
          r2.columnRotation(B);
          r2.columnRotation(V);

          detail::process<0>(B, U, S, V);
          detail::sort_sigma<0>(U, S, V);
        }
        /**
         *  If B is of form
         *      0 x 0
         *      0 x x
         *      0 0 x
         **/
        else if (abs(alpha[0]) <= tol) {
          /**
           *    Reduce B to
           *        0 0 +
           *        0 x x
           *        0 0 x
           **/
          GivensRotation<value_type> r1(0, 1);
          r1.computeUnconventional(B(0, 1), B(1, 1));
          r1.rowRotation(B);
          r1.columnRotation(U);

          /**
           *    Reduce B to
           *        0 0 0
           *        0 x x
           *        0 + x
           **/
          GivensRotation<value_type> r2(0, 2);
          r2.computeUnconventional(B(0, 2), B(2, 2));
          r2.rowRotation(B);
          r2.columnRotation(U);

          detail::process<1>(B, U, S, V);
          detail::sort_sigma<1>(U, S, V);
        }
        return std::make_tuple(U, S, V);
      } else {
      }
    }

    // Polar guarantees negative sign is on the small magnitude singular value.
    // S is guaranteed to be the closest one to identity.
    // R is guaranteed to be the closest rotation to A.
    template <typename VecT,
              enable_if_all<VecT::dim == 2,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto polar_decomposition(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto N = VecT::template range_t<0>::value;
      typename VecT::template variant_vec<value_type, typename VecT::extents> R{};
      if constexpr (N == 1) {
        R(0, 0) = 1;
        return std::make_tuple(R, A.clone());
      } else if constexpr (N == 2) {
        GivensRotation<value_type> r{0, 1};
        auto S = polar_decomposition(A, r);
        r.fill(R);
        return std::make_tuple(R, S);
      } else if constexpr (N == 3) {
        auto [U, S, V] = qr_svd(A);
        R = U * V.transpose();
        auto Ssym = ::zs::diag_mul(V, S) * V.transpose();
        return std::make_tuple(R, Ssym);
      } else {
      }
    }

  }  // namespace math

}  // namespace zs
