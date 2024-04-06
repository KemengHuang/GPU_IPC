/**
Givens rotation
*/
#pragma once
#include <cmath>

#include "../VecInterface.hpp"

namespace zs {

  /// ref: https://www.math.ucla.edu/~cffjiang/research/svd/svd.pdf
  /// ref:
  /// https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Math/Linear/Givens.h
  namespace math {
    /**
     *    Class for givens rotation.
     *    Row rotation G*A corresponds to something like
     *      c -s  0
     *    ( s  c  0 ) A
     *      0  0  1
     *    Column rotation A G' corresponds to something like
     *        c -s  0
     *    A ( s  c  0 )
     *        0  0  1

     *    c and s are always computed so that
     *    ( c -s ) ( a )  =  ( * )
     *      s  c     b       ( 0 )

     *    Assume rowi<rowk.
     **/
    template <typename T> struct GivensRotation {
      /**
       *  [  c  s ]
       *  [ -s  c ]
       */
      int rowi{0};
      int rowk{1};
      T c{1};
      T s{0};

      constexpr GivensRotation() noexcept = default;
      constexpr GivensRotation(int rowi_in, int rowk_in) noexcept
          : rowi{rowi_in}, rowk{rowk_in}, c{1}, s{0} {}
      constexpr GivensRotation(T a, T b, int rowi_in, int rowk_in) noexcept
          : rowi{rowi_in}, rowk{rowk_in} {
        computeConventional(a, b);
      }

      constexpr void setIdentity() noexcept {
        c = 1;
        s = 0;
      }

      constexpr void transposeInPlace() noexcept { s = -s; }

      /**
       *    GivensConventional
       *    Compute c and s from a and b so that
       *    ( c -s ) ( a )  =  ( * )
       *      s  c     b       ( 0 )
       **/
      constexpr void computeConventional(const T a, const T b) {
        T d = a * a + b * b;
        c = 1;
        s = 0;
        if (d != 0) {
          T t{zs::rsqrt(d)};
          c = a * t;
          s = -b * t;
        }
      }

      /**
       *    GivensUnconventional
       *    This function computes c and s so that
       *    ( c -s ) ( a )  =  ( 0 )
       *      s  c     b       ( * )
       **/
      constexpr void computeUnconventional(const T a, const T b) {
        T d = a * a + b * b;
        c = 0;
        s = 1;
        if (d != 0) {
          T t{zs::rsqrt(d)};
          s = a * t;
          c = b * t;
        }
      }

      /**
       *    Fill the R with the entries of this rotation
       **/
      template <typename VecT,
                enable_if_all<VecT::dim == 2,
                              VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                              std::is_floating_point_v<typename VecT::value_type>> = 0>
      constexpr void fill(VecInterface<VecT>& A) const noexcept {
        A.assign(A.identity());
        A(rowi, rowi) = c;
        A(rowk, rowi) = -s;
        A(rowi, rowk) = s;
        A(rowk, rowk) = c;
      }

      /**
       *    This function does something like G^T A -> A
       *    [ c -s  0 ]
       *    [ s  c  0 ] A -> A
       *    [ 0  0  1 ]
       *    It only affects row i and row k of A.
       **/
      template <typename VecT,
                enable_if_all<VecT::dim == 2,
                              VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                              std::is_floating_point_v<typename VecT::value_type>> = 0>
      constexpr void rowRotation(VecInterface<VecT>& A) const noexcept {
        using index_type = typename VecT::index_type;
        constexpr auto ncols = VecT::template range_t<1>::value;
        for (index_type j = 0; j != ncols; ++j) {
          auto tau1 = A(rowi, j);
          auto tau2 = A(rowk, j);
          A(rowi, j) = c * tau1 - s * tau2;
          A(rowk, j) = s * tau1 + c * tau2;
        }
      }
      /**
       *    This function does something like A G -> A
       *        [ c  s  0 ]
       *    A   [-s  c  0 ]  -> A
       *        [ 0  0  1 ]
       *    It only affects column i and column k of A.
       **/
      template <typename VecT,
                enable_if_all<VecT::dim == 2,
                              VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                              std::is_floating_point_v<typename VecT::value_type>> = 0>
      constexpr void columnRotation(VecInterface<VecT>& A) const noexcept {
        using index_type = typename VecT::index_type;
        constexpr auto nrows = VecT::template range_t<0>::value;
        for (index_type i = 0; i != nrows; ++i) {
          auto tau1 = A(i, rowi);
          auto tau2 = A(i, rowk);
          A(i, rowi) = c * tau1 - s * tau2;
          A(i, rowk) = s * tau1 + c * tau2;
        }
      }

      /**
       * the multiplied givens rotation must be for same row and column
       **/
      constexpr void operator*=(const GivensRotation<T>& A) noexcept {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
      }
      /**
       * the multiplied givens rotation must be for same row and column
       **/
      constexpr GivensRotation<T> operator*(const GivensRotation<T>& A) const noexcept {
        GivensRotation<T> r{*this};
        r *= A;
        return r;
      }
    };

    template <typename VecT> static constexpr bool is_3_by_3() noexcept {
      if constexpr (VecT::dim == 2)
        return VecT::template range_t<0>::value == 3 && VecT::template range_t<1>::value == 3;
      return false;
    }
    /**
     * \brief zero chasing the 3X3 matrix to bidiagonal form
     * original form of H:
     * x x 0
     * x x x
     * 0 0 x
     * after zero chase:
     * x x 0
     * 0 x x
     * 0 0 x
     * \note don't follow algorithm 5 psudo code, use ziran2020 implementation
     **/
    template <typename VecH, typename VecU, typename VecV,
              enable_if_all<is_3_by_3<VecH>(), is_3_by_3<VecU>(), is_3_by_3<VecV>()> = 0>
    constexpr void zero_chasing(VecInterface<VecH>& H, VecInterface<VecU>& U,
                                VecInterface<VecV>& V) noexcept {
      using T = typename VecH::value_type;
      /**
          Reduce H to of form
          x x +
          0 x x
          0 0 x
          */
      GivensRotation<T> r1(H(0, 0), H(1, 0), 0, 1);
      /**
          Reduce H to of form
          x x 0
          0 x x
          0 + x
          Can calculate r2 without multiplying by r1 since both entries are in first two
          rows thus no need to divide by sqrt(a^2+b^2)
          */
      GivensRotation<T> r2(1, 2);
      if (H(1, 0) != 0)
        r2.computeConventional(H(0, 0) * H(0, 1) + H(1, 0) * H(1, 1),
                               H(0, 0) * H(0, 2) + H(1, 0) * H(1, 2));
      else
        r2.computeConventional(H(0, 1), H(0, 2));

      r1.rowRotation(H);

      /* GivensRotation<T> r2(H(0, 1), H(0, 2), 1, 2); */
      r2.columnRotation(H);
      r2.columnRotation(V);

      /**
          Reduce H to of form
          x x 0
          0 x x
          0 0 x
          */
      GivensRotation<T> r3(H(1, 1), H(2, 1), 1, 2);
      r3.rowRotation(H);

      // Save this till end for better cache coherency
      // r1.rowRotation(u_transpose);
      // r3.rowRotation(u_transpose);
      r1.columnRotation(U);
      r3.columnRotation(U);
    }
    /**
     *    \brief make a 3X3 matrix to upper bidiagonal form
     *    original form of H:   x x x
     *                          x x x
     *                          x x x
     *    after zero chase:
     *                          x x 0
     *                          0 x x
     *                          0 0 x
     **/
    template <typename VecH, typename VecU, typename VecV>
    constexpr void upper_bidiagonalize(VecInterface<VecH>& H, VecInterface<VecU>& U,
                                       VecInterface<VecV>& V) {
      U.assign(U.identity());
      V.assign(V.identity());

      /**
       *  Reduce H to of form
       *                        x x x
       *                        x x x
       *                        0 x x
       **/
      GivensRotation<typename VecH::value_type> r{H(1, 0), H(2, 0), 1, 2};
      r.rowRotation(H);
      r.columnRotation(U);
      zero_chasing(H, U, V);
    }

    /**
         \brief make a 3X3 matrix to lambda shape
         original form of H:   x x x
         *                     x x x
         *                     x x x
         after :
         *                     x 0 0
         *                     x x 0
         *                     x 0 x
      */
    template <typename VecH, typename VecU, typename VecV,
              enable_if_all<is_3_by_3<VecH>(), is_3_by_3<VecU>(), is_3_by_3<VecV>()> = 0>
    constexpr void make_lambda_shape(VecInterface<VecH>& H, VecInterface<VecU>& U,
                                     VecInterface<VecV>& V) noexcept {
      using T = typename VecH::value_type;
      U.assign(U.identity());
      V.assign(V.identity());

      /**
        Reduce H to of form
        *                    x x 0
        *                    x x x
        *                    x x x
        */

      GivensRotation<T> r1(H(0, 1), H(0, 2), 1, 2);
      r1.columnRotation(H);
      r1.columnRotation(V);

      /**
        Reduce H to of form
        *                    x x 0
        *                    x x 0
        *                    x x x
        */

      r1.computeUnconventional(H(1, 2), H(2, 2));
      r1.rowRotation(H);
      r1.columnRotation(U);

      /**
        Reduce H to of form
        *                    x x 0
        *                    x x 0
        *                    x 0 x
        */

      GivensRotation<T> r2(H(2, 0), H(2, 1), 0, 1);
      r2.columnRotation(H);
      r2.columnRotation(V);

      /**
        Reduce H to of form
        *                    x 0 0
        *                    x x 0
        *                    x 0 x
        */
      r2.computeUnconventional(H(0, 1), H(1, 1));
      r2.rowRotation(H);
      r2.columnRotation(U);
    }

  }  // namespace math

}  // namespace zs
