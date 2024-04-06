#pragma once

#include <cmath>
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
#  include "math.h"  // CUDA math library
#endif
#include <functional>
#include <type_traits>
#include <utility>

#include "Complex.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/meta/ControlFlow.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  namespace mathutil_impl {
    // constexpr scan only available in c++20:
    // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
    template <typename... Args, std::size_t... Is>
    constexpr auto incl_prefix_sum_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is <= I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_prefix_sum_impl(std::size_t I, std::index_sequence<Is...>,
                                        Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is < I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_suffix_mul_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is > I ? std::forward<Args>(args) : 1) * ...);
    }
  }  // namespace mathutil_impl

  /// copied from gcem_options.hpp
  constexpr double g_pi = 3.1415926535897932384626433832795028841972L;
  constexpr double g_half_pi = 1.5707963267948966192313216916397514420986L;
  constexpr double g_sqrt2 = 1.4142135623730950488016887242096980785697L;

  namespace math {

    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr bool near_zero(T v) noexcept {
      constexpr auto eps = (T)128 * limits<T>::epsilon();
      return v >= -eps && v <= eps;
    }

    /// binary_op_result
    template <typename T0, typename T1> struct binary_op_result {
      static constexpr auto determine_type() noexcept {
        if constexpr (std::is_integral_v<T0> && std::is_integral_v<T1>) {
          using bigger_type = conditional_t<(sizeof(T0) >= sizeof(T1)), T0, T1>;
          if constexpr (std::is_signed_v<T0> || std::is_signed_v<T1>)
            return std::make_signed_t<bigger_type>{};
          else
            return bigger_type{};
        } else
          return std::common_type_t<T0, T1>{};
      }
      using type = decltype(determine_type());
    };
    template <typename T0, typename T1> using binary_op_result_t =
        typename binary_op_result<T0, T1>::type;

    /// op_result
    template <typename... Ts> struct op_result;
    template <typename T> struct op_result<T> { using type = T; };
    template <typename T, typename... Ts> struct op_result<T, Ts...> {
      using type = binary_op_result_t<T, typename op_result<Ts...>::type>;
    };
    template <typename... Args> using op_result_t = typename op_result<Args...>::type;
  }  // namespace math

  /**
   *  math intrinsics (not constexpr at all! just cheating the compiler)
   */
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T copysign(T mag, T sgn) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::copysignf(mag, sgn);
    else
      return ::copysign((double)mag, (double)sgn);
#else
    return std::copysign(mag, sgn);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T abs(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fabsf(v);
    else
      return ::fabs((double)v);
#else
    return std::abs(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T max(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmaxf(x, y);
    else
      return ::fmax((double)x, (double)y);
#else
    return std::max(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T min(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fminf(x, y);
    else
      return ::fmin((double)x, (double)y);
#else
    return std::min(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T fma(T x, T y, T z) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmaf(x, y, z);
    else
      return ::fma((double)x, (double)y, (double)z);
#else
    return std::fma(x, y, z);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T fmod(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmodf(x, y);
    else
      return ::fmod((double)x, (double)y);
#else
    return std::fmod(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T ceil(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::ceilf(v);
    else
      return ::ceil((double)v);
#else
    return std::ceil(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T floor(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::floorf(v);
    else
      return ::floor((double)v);
#else
    return std::floor(v);
#endif
  }

  // different from math::sqrt
  template <typename T, enable_if_t<std::is_arithmetic_v<T>> = 0> constexpr T sqr(T v) noexcept {
    return v * v;
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T sqrt(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::sqrtf(v);
    else
      return ::sqrt((double)v);
#else
    return std::sqrt(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T rsqrt(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::rsqrtf(v);
    else
      return ::rsqrt((double)v);
#else
    return (T)1 / (T)std::sqrt(v);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T log(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::logf(v);
    else
      return ::log((double)v);
#else
    return std::log(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T log1p(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::log1pf(v);
    else
      return ::log1p((double)v);
#else
    return std::log1p(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T exp(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::expf(v);
    else
      return ::exp((double)v);
#else
    return std::exp(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T pow(T base, T exp) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::powf(base, exp);
    else
      return ::pow((double)base, (double)exp);
#else
    return std::pow(base, exp);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  ZS_FUNCTION T add_ru(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::__fadd_ru(x, y);
    else
      return ::__dadd_ru((double)x, (double)y);
#else
    return (x + y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  ZS_FUNCTION T sub_ru(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::__fsub_ru(x, y);
    else
      return ::__dsub_ru((double)x, (double)y);
#else
    return (x - y);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T sinh(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::sinhf(v);
    else
      return ::sinh((double)v);
#else
    return std::sinh(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T sin(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::sinf(v);
    else
      return ::sin((double)v);
#else
    return std::sin(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T asinh(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::asinhf(v);
    else
      return ::asinh((double)v);
#else
    return std::asinh(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T asin(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::asinf(v);
    else
      return ::asin((double)v);
#else
    return std::asin(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T cosh(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::coshf(v);
    else
      return ::cosh((double)v);
#else
    return std::cosh(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T cos(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::cosf(v);
    else
      return ::cos((double)v);
#else
    return std::cos(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T acosh(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::acoshf(v);
    else
      return ::acosh((double)v);
#else
    return std::acosh(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T acos(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::acosf(v);
    else
      return ::acos((double)v);
#else
    return std::acos(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T atan2(T y, T x) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::atan2f(y, x);
    else
      return ::atan2((double)y, (double)x);
#else
    return std::atan2(y, x);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr bool isnan(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    return ::isnan(v) != 0;  // due to msvc
#else
    return std::isnan(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr bool isinf(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    return ::isinf(v) != 0;  // due to msvc
#else
    return std::isinf(v);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T modf(T x, T *iptr) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    static_assert(is_same_v<T, float> || is_same_v<T, double>, "modf only supports float/double");
    if constexpr (is_same_v<T, float>)
      return ::modff(x, iptr);
    else if constexpr (is_same_v<T, double>)
      return ::modf(x, iptr);
#else
    return std::modf(x, iptr);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T frexp(T x, int *exp) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::frexpf(x, exp);
    else
      return ::frexp((double)x, exp);
#else
    return std::frexp(x, exp);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T ldexp(T x, int exp) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::ldexpf(x, exp);  // scalbnf(x, exp)
    else
      return ::ldexp((double)x, exp);
#else
    return std::ldexp(x, exp);
#endif
  }

  // 26.2.7/3 abs(__z):  Returns the magnitude of __z.
  template <typename T> constexpr T abs(const complex<T> &z) noexcept {
    T x = z.real();
    T y = z.imag();
    const T s = zs::max(zs::abs(x), zs::abs(y));
    if (s == T{}) return s;
    x /= s;
    y /= s;
    return s * zs::sqrt(x * x + y * y);
  }
  // 26.2.7/4: arg(__z): Returns the phase angle of __z.
  template <typename T> constexpr T arg(const complex<T> &z) noexcept {
    return zs::atan2(z.imag(), z.real());
  }

  template <typename T> constexpr complex<T> polar(const T &rho, const T &theta) {
    // assert(rho >= 0);
    return complex<T>{rho * zs::cos(theta), rho * zs::sin(theta)};
  }

  // 26.2.8/1 cos(__z):  Returns the cosine of __z.
  template <typename T> constexpr complex<T> cos(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::cos(x) * zs::cosh(y), -zs::sin(x) * zs::sinh(y)};
  }
  template <typename T> constexpr complex<T> cosh(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::cosh(x) * zs::cos(y), zs::sinh(x) * zs::sin(y)};
  }

  // 26.2.8/3 exp(__z): Returns the complex base e exponential of x
  template <typename T> constexpr complex<T> exp(const complex<T> &z) {
    return zs::polar(zs::exp(z.real()), z.imag());
  }

  // 26.2.8/5 log(__z): Returns the natural complex logarithm of __z.
  //                    The branch cut is along the negative axis.
  template <typename T> constexpr complex<T> log(const complex<T> &z) {
    return complex<T>{zs::log(zs::abs(z)), zs::arg(z)};
  }
  template <typename T> constexpr complex<T> log10(const complex<T> &z) {
    return zs::log(z) / zs::log((T)10.0);
  }

  // 26.2.8/10 sin(__z): Returns the sine of __z.
  template <typename T> constexpr complex<T> sin(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::sin(x) * zs::cosh(y), zs::cos(x) * zs::sinh(y)};
  }

  // 26.2.8/11 sinh(__z): Returns the hyperbolic sine of __z.
  template <typename T> constexpr complex<T> sinh(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::sinh(x) * zs::cos(y), zs::cosh(x) * zs::sin(y)};
  }

  // 26.2.8/13 sqrt(__z): Returns the complex square root of __z.
  //                     The branch cut is on the negative axis.
  template <typename T> constexpr complex<T> sqrt(const complex<T> &z) {
    T x = z.real();
    T y = z.imag();
    if (x == T{}) {
      T t = zs::sqrt(zs::abs(y) / 2);
      return complex<T>{t, y < T{} ? -t : t};
    } else {
      T t = zs::sqrt(2 * (zs::abs(z) + zs::abs(x)));
      T u = t / 2;
      return x > T{} ? complex<T>{u, y / t} : complex<T>{zs::abs(y) / t, y < T{} ? -u : u};
    }
  }

  // 26.2.8/14 tan(__z):  Return the complex tangent of __z.
  template <typename T> constexpr complex<T> tan(const complex<T> &z) {
    return zs::sin(z) / zs::cos(z);
  }

  // 26.2.8/15 tanh(__z):  Returns the hyperbolic tangent of __z.
  template <typename T> constexpr complex<T> tanh(const complex<T> &z) {
    return zs::sinh(z) / zs::cosh(z);
  }

  namespace detail {
    // 26.2.8/9  pow(__x, __y): Returns the complex power base of __x
    //                          raised to the __y-th power.  The branch
    //                          cut is on the negative axis.
    template <typename T>
    constexpr complex<T> __complex_pow_unsigned(const complex<T> &x, unsigned n) {
      complex<T> y = n % 2 ? x : complex<T>(1);

      while (n >>= 1) {
        x *= x;
        if (n % 2) y *= x;
      }
      return y;
    }
  }  // namespace detail

  // In C++11 mode we used to implement the resolution of
  // DR 844. complex pow return type is ambiguous.
  // thus the following overload was disabled in that mode.  However, doing
  // that causes all sorts of issues, see, for example:
  //   http://gcc.gnu.org/ml/libstdc++/2013-01/msg00058.html
  // and also PR57974.
  template <typename T> constexpr complex<T> pow(const complex<T> &z, int n) {
    return n < 0 ? complex<T>{1} / detail::__complex_pow_unsigned(z, (unsigned)-n)
                 : detail::__complex_pow_unsigned(z, n);
  }

  template <typename T> constexpr complex<T> pow(const complex<T> &x, const complex<T> &y) {
    return x == T{} ? T{} : zs::exp(y * zs::log(x));
  }

  template <typename T, enable_if_t<!std::is_integral_v<T>> = 0>
  constexpr complex<T> pow(const complex<T> &x, const T &y) {
    if (x.imag() == T{} && x.real() > T{}) return zs::pow(x.real(), y);

    complex<T> t = zs::log(x);
    return zs::polar<T>(zs::exp(y * t.real()), y * t.imag());
  }

  template <typename T> constexpr complex<T> pow(const T &x, const complex<T> &y) {
    return x > T{} ? zs::polar<T>(zs::pow(x, y.real()), y.imag() * zs::log(x))
                   : zs::pow(complex<T>{x}, y);
  }

  /// acos(__z) [8.1.2].
  //  Effects:  Behaves the same as C99 function cacos, defined
  //            in subclause 7.3.5.1.
  template <typename T> constexpr complex<T> acos(const complex<T> &z) {
    const complex<T> t = zs::asin(z);
    const T __pi_2 = 1.5707963267948966192313216916397514L;
    return complex<T>{__pi_2 - t.real(), -t.imag()};
  }
  /// asin(__z) [8.1.3].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.5.2.
  template <typename T> constexpr complex<T> asin(const complex<T> &z) {
    complex<T> t{-z.imag(), z.real()};
    t = zs::asinh(t);
    return complex<T>{t.imag(), -t.real()};
  }
  /// atan(__z) [8.1.4].
  //  Effects:  Behaves the same as C99 function catan, defined
  //            in subclause 7.3.5.3.
  template <typename T> constexpr complex<T> atan(const complex<T> &z) {
    const T r2 = z.real() * z.real();
    const T x = (T)1.0 - r2 - z.imag() * z.imag();

    T num = z.imag() + (T)1.0;
    T den = z.imag() - (T)1.0;

    num = r2 + num * num;
    den = r2 + den * den;

    return complex<T>{(T)0.5 * zs::atan2((T)2.0 * z.real(), x), (T)0.25 * zs::log(num / den)};
  }
  /// acosh(__z) [8.1.5].
  //  Effects:  Behaves the same as C99 function cacosh, defined
  //            in subclause 7.3.6.1.
  template <typename T> constexpr complex<T> acosh(const complex<T> &z) {
    // Kahan's formula.
    return (T)2.0 * zs::log(zs::sqrt((T)0.5 * (z + (T)1.0)) + zs::sqrt((T)0.5 * (z - (T)1.0)));
  }
  /// asinh(__z) [8.1.6].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.6.2.
  template <typename T> constexpr complex<T> asinh(const complex<T> &z) {
    complex<T> t{(z.real() - z.imag()) * (z.real() + z.imag()) + (T)1.0,
                 (T)2.0 * z.real() * z.imag()};
    t = zs::sqrt(t);
    return zs::log(t + z);
  }
  /// atanh(__z) [8.1.7].
  //  Effects:  Behaves the same as C99 function catanh, defined
  //            in subclause 7.3.6.3.
  template <typename T> constexpr complex<T> atanh(const complex<T> &z) {
    const T i2 = z.imag() * z.imag();
    const T x = T(1.0) - i2 - z.real() * z.real();

    T num = T(1.0) + z.real();
    T den = T(1.0) - z.real();

    num = i2 + num * num;
    den = i2 + den * den;

    return complex<T>{(T)0.25 * (zs::log(num) - zs::log(den)),
                      (T)0.5 * zs::atan2((T)2.0 * z.imag(), x)};
  }
#if 0
  /// fabs(__z) [8.1.8].
  //  Effects:  Behaves the same as C99 function cabs, defined
  //            in subclause 7.3.8.1.
  template <typename T> constexpr T fabs(const complex<T> &z) { return zs::abs(z); }
#endif

  /// additional overloads [8.1.9]
  template <typename T> constexpr auto arg(T x) {
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "invalid param type for func [arg]");
    using type = conditional_t<std::is_floating_point_v<T>, T, double>;
    return zs::arg(complex<type>{x});
  }
  // ignore the remaining type promotions for now

  /// customized zpc calls
  namespace math {
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
    constexpr T min(T x, T y) noexcept {
      return y < x ? y : x;
    }
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
    constexpr T max(T x, T y) noexcept {
      return y > x ? y : x;
    }
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0> constexpr T abs(T x) noexcept {
      return x < 0 ? -x : x;
    }
    // TODO refer to:
    // https://github.com/mountunion/ModGCD-OneGPU/blob/master/ModGCD-OneGPU.pdf
    // http://www.iaeng.org/IJCS/issues_v42/issue_4/IJCS_42_4_01.pdf
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr Ti gcd(Ti u, Ti v) noexcept {
      while (v != 0) {
        auto r = u % v;
        u = v;
        v = r;
      }
      return u;
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr Ti lcm(Ti u, Ti v) noexcept {
      return (u / gcd(u, v)) * v;
    }

#if 0
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T x, T curr = 1, T prev = 0) noexcept {
      return curr == prev ? curr : sqrtNewtonRaphson(x, (T)0.5 * (curr + x / curr), curr);
    }
#else
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T n, T relTol = (T)(sizeof(T) > 4 ? 1e-9 : 1e-6)) noexcept {
      constexpr auto eps = (T)128 * limits<T>::epsilon();
      if (n < -eps) return (T)limits<T>::quiet_NaN();
      if (n < (T)eps) return (T)0;

      T xn = (T)1;
      T xnp1 = (T)0.5 * (xn + n / xn);
      for (const auto tol = max(n * relTol, eps); abs(xnp1 - xn) > tol;
           xnp1 = (T)0.5 * (xn + n / xn))
        xn = xnp1;
      return xnp1;
    }
#endif
    /// ref: http://www.lomont.org/papers/2003/InvSqrt.pdf
    /// ref: https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    /// ref: https://community.wolfram.com/groups/-/m/t/1108896
    /// ref:
    /// https://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi
    constexpr float q_rsqrt(float number) noexcept {
      uint32_t i{};
      float x2 = number * 0.5f, y = number;
      // i = *(uint32_t *)&y;
      i = reinterpret_bits<uint32_t>(y);
      i = 0x5f375a86 - (i >> 1);
      // y = *(float *)&i;
      y = reinterpret_bits<float>(i);
      y = y * (1.5f - (x2 * y * y));
      y = y * (1.5f - (x2 * y * y));
      return y;
    }
    constexpr double q_rsqrt(double number) noexcept {
      uint64_t i{};
      double x2 = number * 0.5, y = number;
      // i = *(uint64_t *)&y;
      i = reinterpret_bits<uint64_t>(y);
      i = 0x5fe6eb50c7b537a9 - (i >> 1);
      // y = *(double *)&i;
      y = reinterpret_bits<double>(i);
      y = y * (1.5 - (x2 * y * y));
      y = y * (1.5 - (x2 * y * y));
      return y;
    }
    constexpr float q_sqrt(float x) noexcept { return 1.f / q_rsqrt(x); }
    // best guess starting square
    constexpr double q_sqrt(double fp) noexcept { return 1.0 / q_rsqrt(fp); }
    /// ref:
    /// https://stackoverflow.com/questions/66752842/ieee-754-conformant-sqrtf-implementation-taking-into-account-hardware-restrict
    /* square root computation suitable for all IEEE-754 binary32 arguments */
    constexpr float sqrt(float arg) noexcept {
      const float FP32_INFINITY = reinterpret_bits<float>(0x7f800000u);
      const float FP32_QNAN = reinterpret_bits<float>(0x7fffffffu); /* system specific */
      const float scale_in = 0x1.0p+26f;
      const float scale_out = 0x1.0p-13f;
      float rsq{}, err{}, sqt{};

      if (arg < 0.0f) {
        return FP32_QNAN;
      }
      // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1c6fe34b4ac091e40eceeb0bae58459f
      else if ((arg == 0.0f) || !(abs(arg) < FP32_INFINITY)) { /* Inf, NaN */
        return arg + arg;
      } else {
        /* scale subnormal arguments towards unity */
        arg = arg * scale_in;

        /* generate low-accuracy approximation to rsqrt(arg) */
        rsq = q_rsqrt(arg);

        /* apply two Newton-Raphson iterations with quadratic convergence */
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;

        /* compute sqrt from rsqrt, round to nearest or even */
        sqt = rsq * arg;
        err = sqt * -sqt + arg;
        sqt = (0.5f * rsq * err + sqt);

        /* compensate scaling of argument by counter-scaling the result */
        sqt = sqt * scale_out;

        return sqt;
      }
    }

    namespace detail {
      template <typename T, typename Tn, enable_if_t<std::is_integral_v<Tn>> = 0>
      constexpr T pow_integral_recursive(T base, T val, Tn exp) noexcept {
        return exp > (Tn)1
                   ? ((exp & (Tn)1) ? pow_integral_recursive(base * base, val * base, exp / (Tn)2)
                                    : pow_integral_recursive(base * base, val, exp / (Tn)2))
                   : (exp == (Tn)1 ? val * base : val);
      }
    }  // namespace detail
    template <typename T, typename Tn,
              enable_if_all<std::is_arithmetic_v<T>, std::is_integral_v<Tn>> = 0>
    constexpr auto pow_integral(T base, Tn exp) noexcept {
      using R = T;  // math::op_result_t<T0, T1>;
      return exp == (Tn)3
                 ? base * base * base
                 : (exp == (Tn)2
                        ? base * base
                        : (exp == (Tn)1
                               ? base
                               : (exp == (Tn)0 ? (R)1
                                               : (exp == limits<Tn>::max()
                                                      ? limits<R>::infinity()
                                                      // make signed to get rid of compiler warn
                                                      : ((std::make_signed_t<Tn>)exp < 0
                                                             ? (R)0
                                                             : detail::pow_integral_recursive(
                                                                 (R)base, (R)1, (Tn)exp))))));
    }

    /**
     * Robustly computing log(x+1)/x
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T log_1px_over_x(const T x, const T eps = 1e-6) noexcept {
      if (abs(x) < eps)
        return (T)1 - x / (T)2 + x * x / (T)3 - x * x * x / (T)4;
      else {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
        if constexpr (is_same_v<T, float>)
          return ::log1pf(x) / x;
        else
          return ::log1p(x) / x;
#else
        if constexpr (is_same_v<T, float>)
          return std::log1pf(x) / x;
        else
          return std::log1p(x) / x;
#endif
      }
    }
    /**
     * Robustly computing (logx-logy)/(x-y)
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T diff_log_over_diff(const T x, const T y, const T eps = 1e-6) noexcept {
      return log_1px_over_x(x / y - (T)1, eps) / y;
    }
    /**
     * Robustly computing (x logy- y logx)/(x-y)
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T diff_interlock_log_over_diff(const T x, const T y, const T logy,
                                             const T eps = 1e-6) noexcept {
      return logy - y * diff_log_over_diff(x, y, eps);
    }

    /// ref: codim-IPC
    template <typename T> constexpr T get_smallest_positive_real_quad_root(T a, T b, T c, T tol) {
      // return negative value if no positive real root is found
      T t{};
      if (zs::abs(a) <= tol)
        t = -c / b;
      else {
        double desc = b * b - 4 * a * c;
        if (desc > 0) {
          t = (-b - zs::sqrt(desc)) / (2 * a);
          if (t < 0) t = (-b + zs::sqrt(desc)) / (2 * a);
        } else  // desv<0 ==> imag
          t = -1;
      }
      return t;
    }
    template <typename T>
    constexpr T get_smallest_positive_real_cubic_root(T a, T b, T c, T d, T tol) {
      // return negative value if no positive real root is found
      T t = -1;
      if (zs::abs(a) <= tol)
        t = get_smallest_positive_real_quad_root(b, c, d, tol);
      else {
        zs::complex<T> i(0, 1);
        zs::complex<T> delta0(b * b - 3 * a * c, 0);
        zs::complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        zs::complex<T> C = zs::pow(
            (delta1 + zs::sqrt(delta1 * delta1 - (T)4.0 * delta0 * delta0 * delta0)) / (T)2.0,
            (T)1.0 / (T)3.0);
        if (zs::abs(C) == (T)0.0) {
          // a corner case listed by wikipedia found by our collaborate from another project
          C = zs::pow(
              (delta1 - zs::sqrt(delta1 * delta1 - (T)4.0 * delta0 * delta0 * delta0)) / (T)2.0,
              (T)1.0 / (T)3.0);
        }
        zs::complex<T> u2 = ((T)-1.0 + zs::sqrt((T)3.0) * i) / (T)2.0;
        zs::complex<T> u3 = ((T)-1.0 - zs::sqrt((T)3.0) * i) / (T)2.0;
        zs::complex<T> t1 = (b + C + delta0 / C) / ((T)-3.0 * a);
        zs::complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / ((T)-3.0 * a);
        zs::complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / ((T)-3.0 * a);

        if ((zs::abs(imag(t1)) < tol) && (real(t1) > 0)) t = real(t1);
        if ((zs::abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
          t = real(t2);
        if ((zs::abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
          t = real(t3);
      }
      return t;
    }

    template <typename T> constexpr T newton_solve_for_cubic_equation(T a, T b, T c, T d,
                                                                      T *results, int &numSols,
                                                                      T eps) {
      const auto __f
          = [](T x, T a, T b, T c, T d) { return a * x * x * x + b * x * x + c * x + d; };
      const auto __df = [](T x, T a, T b, T c) { return 3 * a * x * x + 2 * b * x + c; };
      T DX = 0;
      numSols = 0;
      T specialPoint = -b / a / 3;
      T pos[2] = {};
      int solves = 1;
      T delta = 4 * b * b - 12 * a * c;
      if (delta > 0) {
        pos[0] = (zs::sqrt(delta) - 2 * b) / 6 / a;
        pos[1] = (-zs::sqrt(delta) - 2 * b) / 6 / a;
        T v1 = __f(pos[0], a, b, c, d);
        T v2 = __f(pos[1], a, b, c, d);
        if (zs::abs(v1) < eps * eps) {
          v1 = 0;
        }
        if (zs::abs(v2) < eps * eps) {
          v2 = 0;
        }
        T sign = v1 * v2;
        DX = (pos[0] - pos[1]);
        if (sign <= 0) {
          solves = 3;
        } else if (sign > 0) {
          if ((a < 0 && __f(pos[0], a, b, c, d) > 0) || (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
            DX = -DX;
          }
        }
      } else if (delta == 0) {
        if (zs::abs(__f(specialPoint, a, b, c, d)) < eps * eps) {
          for (int i = 0; i < 3; i++) {
            T tempReuslt = specialPoint;
            results[numSols] = tempReuslt;
            numSols++;
          }
          return;
        }
        if (a > 0) {
          if (__f(specialPoint, a, b, c, d) > 0) {
            DX = 1;
          } else if (__f(specialPoint, a, b, c, d) < 0) {
            DX = -1;
          }
        } else if (a < 0) {
          if (__f(specialPoint, a, b, c, d) > 0) {
            DX = -1;
          } else if (__f(specialPoint, a, b, c, d) < 0) {
            DX = 1;
          }
        }
      }

      T start = specialPoint - DX;
      T x0 = start;

      for (int i = 0; i < solves; i++) {
        T x1 = 0;
        int itCount = 0;
        do {
          if (itCount) x0 = x1;

          x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
          itCount++;
        } while (zs::abs(x1 - x0) > eps && itCount < 100000);
        results[numSols] = (x1);
        numSols++;
        start = start + DX;
        x0 = start;
      }
    }

  }  // namespace math

  template <typename... Args>
  constexpr auto incl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::incl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_suffix_mul(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_suffix_mul_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto incl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return incl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_suffix_mul(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_suffix_mul(I, Ns...);
  }

  template <typename T, typename Data, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr auto linear_interop(T alpha, Data &&a, Data &&b) noexcept {
    return a + (b - a) * alpha;
  }

  template <typename T, enable_if_t<std::is_integral_v<T>> = 0>
  constexpr auto lower_trunc(const T v) noexcept {
    return v;
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr auto lower_trunc(const T v) noexcept {
    return (conditional_t<sizeof(T) >= sizeof(f64), i64, i32>)zs::floor(v);
  }
  template <
      typename Ti, typename T,
      enable_if_all<std::is_floating_point_v<T>, std::is_integral_v<Ti>, std::is_signed_v<Ti>> = 0>
  constexpr auto lower_trunc(wrapt<Ti>, const T v) noexcept {
    return static_cast<Ti>(zs::floor(v));
  }

}  // namespace zs