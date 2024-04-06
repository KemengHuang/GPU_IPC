#pragma once
#include "Distance.hpp"
#include "zensim/math/VecInterface.hpp"

namespace zs {

  //! point-triangle
  // David Eberly, Geometric Tools, Redmond WA 98052
  // Copyright (c) 1998-2022
  // Distributed under the Boost Software License, Version 1.0.
  // https://www.boost.org/LICENSE_1_0.txt
  // https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
  // Version: 6.0.2022.01.06

  // ref: https://www.geometrictools.com/GTE/Mathematics/DistPointTriangle.h
  template <
      typename VecTP, typename VecTT,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTT::dims>> = 0>
  constexpr auto dist_pt_sqr(const VecInterface<VecTP> &p, const VecInterface<VecTT> &t0,
                             const VecInterface<VecTT> &t1,
                             const VecInterface<VecTT> &t2) noexcept {
    using T = math::op_result_t<typename VecTP::value_type, typename VecTT::value_type>;
    static_assert(std::is_floating_point_v<T>,
                  "value_types of VecTs cannot be both integral type.");
    using TV = typename VecTP::template variant_vec<T, typename VecTP::extents>;
    TV diff = t0 - p;
    TV e0 = t1 - t0;
    TV e1 = t2 - t0;
    T a00 = e0.dot(e0);
    T a01 = e0.dot(e1);
    T a11 = e1.dot(e1);
    T b0 = diff.dot(e0);
    T b1 = diff.dot(e1);
    T det = zs::max(a00 * a11 - a01 * a01, (T)0);
    T s = a01 * b1 - a11 * b0;
    T t = a01 * b0 - a00 * b1;

    if (s + t <= det) {
      if (s < (T)0) {
        if (t < (T)0) {  // region 4
          if (b0 < (T)0) {
            t = (T)0;
            if (-b0 >= a00)
              s = (T)1;
            else
              s = -b0 / a00;
          } else {
            s = (T)0;
            if (b1 >= (T)0)
              t = (T)0;
            else if (-b1 >= a11)
              t = (T)1;
            else
              t = -b1 / a11;
          }
        } else {  // region 3
          s = (T)0;
          if (b1 >= (T)0)
            t = (T)0;
          else if (-b1 >= a11)
            t = (T)1;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) {  // region 5
        t = (T)0;
        if (b0 >= (T)0)
          s = (T)0;
        else if (-b0 >= a00)
          s = (T)1;
        else
          s = -b0 / a00;
      } else {  // region 0
                // minimum at interior point
        s /= det;
        t /= det;
      }
    } else {
      T tmp0{}, tmp1{}, numer{}, denom{};
      if (s < (T)0) {  // region 2
        tmp0 = a01 + b0;
        tmp1 = a11 + b1;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        } else {
          s = (T)0;
          if (tmp1 <= (T)0)
            t = (T)1;
          else if (b1 >= (T)0)
            t = (T)0;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) {  // region 6
        tmp0 = a01 + b1;
        tmp1 = a00 + b0;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            t = (T)1;
            s = (T)0;
          } else {
            t = numer / denom;
            s = (T)1 - t;
          }
        } else {
          t = (T)0;
          if (tmp1 <= (T)0)
            s = (T)1;
          else if (b0 >= (T)0)
            s = (T)0;
          else
            s = -b0 / a00;
        }
      } else {  // region 1
        numer = a11 + b1 - a01 - b0;
        if (numer <= (T)0) {
          s = (T)0;
          t = (T)1;
        } else {
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        }
      }
    }
    auto hitpoint = t0 + s * e0 + t * e1;
    return (p - hitpoint).l2NormSqr();
  }
  template <
      typename VecTP, typename VecTT,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTT::dims>> = 0>
  constexpr auto dist_pt(const VecInterface<VecTP> &p, const VecInterface<VecTT> &t0,
                         const VecInterface<VecTT> &t1, const VecInterface<VecTT> &t2) noexcept {
    return zs::sqrt(dist_pt_sqr(p, t0, t1, t2));
  }

  // edge-edge
  // ref: <<practical geometry algorithms>> - Daniel Sunday
  // ref: http://geomalgorithms.com/a07-_distance.html
  // ref: dist3D_Segment_to_Segment()
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_ee_sqr(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                             const VecInterface<VecTB> &eb0,
                             const VecInterface<VecTB> &eb1) noexcept {
    using T = math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    auto u = ea1 - ea0;
    auto v = eb1 - eb0;
    auto w = ea0 - eb0;
    float a = u.dot(u);  // >= 0
    float b = u.dot(v);
    float c = v.dot(v);  // >= 0
    float d = u.dot(w);
    float e = v.dot(w);
    float D = a * c - b * b;   // >= 0
    float sc{}, sN{}, sD = D;  // sc = sN/sD
    float tc{}, tN{}, tD = D;

    constexpr auto eps = (T)128 * limits<T>::epsilon();
    if (D < eps) {
      sN = (T)0;
      sD = (T)1;
      tN = e;
      tD = c;
    } else {  // get the closest points on the infinite lines
      sN = b * e - c * d;
      tN = a * e - b * d;
      if (sN < (T)0) {
        sN = (T)0;
        tN = e;
        tD = c;
      } else if (sN > sD) {
        sN = sD;
        tN = e + b;
        tD = c;
      }
    }

    if (tN < (T)0) {
      tN = (T)0;
      if (auto _d = -d; _d < (T)0)
        sN = (T)0;
      else if (_d > a)
        sN = sD;
      else {
        sN = _d;
        sD = a;
      }
    } else if (tN > tD) {
      tN = tD;
      if (auto b_d = -d + b; b_d < (T)0)
        sN = (T)0;
      else if (b_d > a)
        sN = sD;
      else {
        sN = b_d;
        sD = a;
      }
    }

    sc = zs::abs(sN) < eps ? (T)0 : sN / sD;
    tc = zs::abs(tN) < eps ? (T)0 : tN / tD;

    auto dP = w + (sc * u) - (tc * v);
    return dP.l2NormSqr();
  }
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_ee(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                         const VecInterface<VecTB> &eb0, const VecInterface<VecTB> &eb1) noexcept {
    return zs::sqrt(dist_ee_sqr(ea0, ea1, eb0, eb1));
  }

  template <typename T> constexpr T barrier(const T d2, const T dHat2, const T kappa) {
#if 0
    T e = 0;
    if (d2 < dHat2) {
      T t2 = d2 - dHat2;
      e = -kappa * (t2 / dHat2) * (t2 / dHat2) * zs::log(d2 / dHat2);
    }
    return e;
#else
    if (d2 >= dHat2) return 0;
    return -kappa * (d2 - dHat2) * (d2 - dHat2) * zs::log(d2 / dHat2);
#endif
  }

  template <typename T> constexpr T barrier_gradient(const T d2, const T dHat2, const T kappa) {
#if 0
    T grad = 0;
    if (d2 < dHat2) {
      T t2 = d2 - dHat2;
      grad = kappa
             * ((t2 / dHat2) * zs::log(d2 / dHat2) * (T)-2 / dHat2
                - ((t2 / dHat2) * (t2 / dHat2)) / d2);
    }
    return grad;
#else
    if (d2 >= dHat2) return 0;
    T t2 = d2 - dHat2;
    return kappa * (t2 * zs::log(d2 / dHat2) * -2 - (t2 * t2) / d2);
#endif
  }

  template <typename T> constexpr T barrier_hessian(const T d2, const T dHat2, const T kappa) {
#if 0
    T hess = 0;
    if (d2 < dHat2) {
      T t2 = d2 - dHat2;
      hess = kappa
             * ((zs::log(d2 / dHat2) * (T)-2.0 - t2 * (T)4.0 / d2) / (dHat2 * dHat2)
                + 1.0 / (d2 * d2) * (t2 / dHat2) * (t2 / dHat2));
    }
    return hess;
#else
    if (d2 >= dHat2) return 0;
    T t2 = d2 - dHat2;
    return kappa * ((zs::log(d2 / dHat2) * -2 - t2 * 4 / d2) + (t2 / d2) * (t2 / d2));
#endif
  }

}  // namespace zs