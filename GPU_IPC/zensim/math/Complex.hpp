#pragma once
/// copied gcc10 library and modified
/// following is its license

// The template and inlines for the -*- C++ -*- complex number classes.

// Copyright (C) 1997-2020 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file include/complex
 *  This is a Standard C++ Library header.
 */

//
// ISO C++ 14882: 26.2  Complex Numbers
// Note: this is not a conforming implementation.
// Initially implemented by Ulrich Drepper <drepper@cygnus.com>
// Improved by Gabriel Dos Reis <dosreis@cmla.ens-cachan.fr>
//

namespace zs {

  template <typename T> struct complex {
    using value_type = T;
    constexpr complex(const T& r = T{}, const T& i = T{}) : _real{r}, _imag{i} {}
    constexpr complex(const complex&) = default;
    constexpr complex& operator=(const complex&) = default;

    template <typename U> constexpr complex(const complex<U>& o)
        : _real{o.real()}, _imag{o.imag()} {}

    constexpr T real() const noexcept { return _real; }
    constexpr T imag() const noexcept { return _imag; }

    constexpr void real(T val) const noexcept { _real = val; }
    constexpr void imag(T val) const noexcept { _imag = val; }

    constexpr complex<T>& operator=(const T& t) {
      _real = t;
      _imag = T{};
      return *this;
    }
    constexpr complex<T>& operator+=(const T& t) {
      _real += t;
      return *this;
    }
    constexpr complex<T>& operator-=(const T& t) {
      _real -= t;
      return *this;
    }
    constexpr complex<T>& operator*=(const T& t) {
      _real *= t;
      _imag *= t;
      return *this;
    }
    constexpr complex<T>& operator/=(const T& t) {
      _real /= t;
      _imag /= t;
      return *this;
    }

    template <typename U> constexpr complex<T>& operator=(const complex<U>& o) {
      _real = o.real();
      _imag = o.imag();
      return *this;
    }
    template <typename U> constexpr complex<T>& operator+=(const complex<U>& o) {
      _real += o.real();
      _imag += o.imag();
      return *this;
    }
    template <typename U> constexpr complex<T>& operator-=(const complex<U>& o) {
      _real -= o.real();
      _imag -= o.imag();
      return *this;
    }
    template <typename U> constexpr complex<T>& operator*=(const complex<U>& o) {
      _real = _real * o.real() - _imag * o.imag();
      _imag = _real * o.imag() + _imag * o.real();
      return *this;
    }
    template <typename U> constexpr complex<T>& operator/=(const complex<U>& o) {
      const T __r = _real * o.real() + _imag * o.imag();
      const T __n = o.real() * o.real() + o.imag() * o.imag();
      _imag = (_imag * o.real() - _real * o.imag()) / __n;
      _real = __r / __n;
      return *this;
    }

    value_type _real, _imag;
  };

  //!@name Binary operators
  // scalar
  template <typename T>
  constexpr auto operator+(const complex<T>& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r += y;
    return r;
  }
  template <typename T> constexpr auto operator+(const complex<T>& x, const T& y) noexcept {
    complex<T> r = x;
    r += y;
    return r;
  }
  template <typename T> constexpr auto operator+(const T& x, const complex<T>& y) noexcept {
    complex<T> r = y;
    r += x;
    return r;
  }

  template <typename T>
  constexpr auto operator-(const complex<T>& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r -= y;
    return r;
  }
  template <typename T> constexpr auto operator-(const complex<T>& x, const T& y) noexcept {
    complex<T> r = x;
    r -= y;
    return r;
  }
  template <typename T> constexpr auto operator-(const T& x, const complex<T>& y) noexcept {
    complex<T> r = -y;  //
    r += x;
    return r;
  }

  template <typename T>
  constexpr auto operator*(const complex<T>& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r *= y;
    return r;
  }
  template <typename T> constexpr auto operator*(const complex<T>& x, const T& y) noexcept {
    complex<T> r = x;
    r *= y;
    return r;
  }
  template <typename T> constexpr auto operator*(const T& x, const complex<T>& y) noexcept {
    complex<T> r = y;
    r *= x;
    return r;
  }

  template <typename T>
  constexpr auto operator/(const complex<T>& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r /= y;
    return r;
  }
  template <typename T> constexpr auto operator/(const complex<T>& x, const T& y) noexcept {
    complex<T> r = x;
    r /= y;
    return r;
  }
  template <typename T> constexpr auto operator/(const T& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r /= y;
    return r;
  }

  template <typename T> constexpr auto operator+(const complex<T>& x) noexcept { return x; }
  template <typename T> constexpr auto operator-(const complex<T>& x) noexcept {
    return complex<T>{-x.real(), -x.imag()};
  }
  // ==
  template <typename T>
  constexpr bool operator==(const complex<T>& x, const complex<T>& y) noexcept {
    return x.real() == y.real() && x.imag() == y.imag();
  }
  template <typename T> constexpr bool operator==(const complex<T>& x, const T& y) noexcept {
    return x.real() == y && x.imag() == T{};
  }
  template <typename T> constexpr bool operator==(const T& x, const complex<T>& y) noexcept {
    return y.real() == x && y.imag() == T{};
  }
  // !=
  template <typename T>
  constexpr bool operator!=(const complex<T>& x, const complex<T>& y) noexcept {
    return x.real() != y.real() || x.imag() != y.imag();
  }
  template <typename T> constexpr bool operator!=(const complex<T>& x, const T& y) noexcept {
    return x.real() != y || x.imag() != T{};
  }
  template <typename T> constexpr bool operator!=(const T& x, const complex<T>& y) noexcept {
    return y.real() != x || y.imag() != T{};
  }

  template <typename T> constexpr T real(const complex<T>& z) noexcept { return z.real(); }
  template <typename T> constexpr T imag(const complex<T>& z) noexcept { return z.imag(); }

  // 26.2.7/5: norm(__z) returns the squared magnitude of __z.
  //     As defined, norm() is -not- a norm is the common mathematical
  //     sense used in numerics.  The helper class _Norm_helper<> tries to
  //     distinguish between builtin floating point and the rest, so as
  //     to deliver an answer as close as possible to the real value.
  template <typename T> constexpr T norm(const complex<T>& z) noexcept {
    return z.real() * z.real() + z.imag() * z.imag();
  }
  template <typename T> constexpr complex<T> conj(const complex<T>& z) noexcept {
    return complex<T>{z.real(), -z.imag()};
  }
#if 0
  //!@name Binary operators
  // scalar
  template <typename T>
  constexpr auto operator OP(const complex<T>& x, const complex<T>& y) noexcept {
    complex<T> r = x;
    r OP = y;
    return r;
  }
  template <typename T> constexpr auto operator OP(const complex<T>& x, const T& y) noexcept {
    complex<T> r = x;
    r OP = y;
    return r;
  }
  template <typename T> constexpr auto operator OP(const T& x, const complex<T>& y) noexcept {
    complex<T> r = y;
    r OP = x;
    return r;
  }

  DEFINE_COMPLEX_OP_SCALAR(+)
  DEFINE_COMPLEX_OP_SCALAR(-)
  DEFINE_COMPLEX_OP_SCALAR(*)
  DEFINE_COMPLEX_OP_SCALAR(/)
#endif

}  // namespace zs