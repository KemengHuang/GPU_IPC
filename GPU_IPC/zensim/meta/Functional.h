#pragma once

#include <functional>
#include <limits>

#include "Meta.h"
#include "Relationship.h"
// #include "zensim/zpc_tpls/tl/function_ref.hpp"

namespace zs {

  // gcem alike, shorter alias for std::numeric_limits
  template <typename T> using limits = std::numeric_limits<T>;

  /// WIP: supplement
  template <template <class...> class Function, typename Oprand> struct map {
    using type = Function<Oprand>;
  };
  template <template <class...> class Function, template <class...> class Functor, typename... Args>
  struct map<Function, Functor<Args...>> {
    using type = Functor<Function<Args>...>;
  };
  template <template <class...> class Function, typename Functor> using map_t =
      typename map<Function, Functor>::type;

  template <typename MapperF, typename Oprand, bool recursive = true> struct map_op {
    using type = decltype(std::declval<MapperF &>()(std::declval<Oprand>()));
  };
  template <typename MapperF, template <class...> class Functor, typename... Args>
  struct map_op<MapperF, Functor<Args...>, true> {
    using type = Functor<typename map_op<MapperF, Args, false>::type...>;
  };
  template <typename MapperF, typename Functor> using map_op_t =
      typename map_op<MapperF, Functor>::type;

  // applicative functor: pure, apply
  // either, apply, join, bind, mcombine, fold

  /// binary operation
  template <typename T = void> using plus = std::plus<T>;
  template <typename T = void> using minus = std::minus<T>;
  template <typename T = void> using logical_or = std::logical_or<T>;
  template <typename T = void> using logical_and = std::logical_and<T>;
  template <typename T = void> using multiplies = std::multiplies<T>;
  template <typename T = void> struct getmax {
    template <typename Auto, typename TT = T, enable_if_t<is_same_v<TT, void>> = 0>
    constexpr Auto operator()(const Auto &lhs, const Auto &rhs) const noexcept {
      return lhs > rhs ? lhs : rhs;
    }
    template <typename TT = T, enable_if_t<!is_same_v<TT, void>> = 0>
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs > rhs ? lhs : rhs;
    }
  };
  template <typename T> struct getmin {
    template <typename Auto, typename TT = T, enable_if_t<is_same_v<TT, void>> = 0>
    constexpr Auto operator()(const Auto &lhs, const Auto &rhs) const noexcept {
      return lhs < rhs ? lhs : rhs;
    }
    template <typename TT = T, enable_if_t<!is_same_v<TT, void>> = 0>
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs < rhs ? lhs : rhs;
    }
  };

  struct static_plus {
    template <typename T = int> constexpr auto e() const noexcept {
      if constexpr (std::is_arithmetic_v<T>)
        return (T)0;
      else
        return 0;
    }
    template <typename... Args> constexpr auto operator()(Args &&...args) const noexcept {
      if constexpr (sizeof...(Args) == 0)
        return e();
      else  // default
        return (FWD(args) + ...);
    }
  };
  struct static_multiplies {
    template <typename T = int> constexpr auto e() const noexcept {
      if constexpr (std::is_arithmetic_v<T>)
        return (T)1;
      else  // default
        return 1;
    }
    template <typename... Args> constexpr auto operator()(Args &&...args) const noexcept {
      if constexpr (sizeof...(Args) == 0)
        return e();
      else
        return (FWD(args) * ...);
    }
  };
  struct static_minus {
    template <typename TA, typename TB> constexpr auto operator()(TA a, TB b) const noexcept {
      return a - b;
    }
  };
  template <bool SafeMeasure = false> struct static_divides {
    template <typename TA, typename TB> constexpr auto operator()(TA a, TB b) const noexcept {
      if constexpr (std::is_arithmetic_v<TB>) {
        if constexpr (SafeMeasure) {
          constexpr auto eps = (TB)128 * limits<TB>::epsilon();
          return (b >= -eps && b <= eps) ? a : a / b;
        } else
          return a / b;
      } else
        return a / b;
    }
  };

  /// monoid operation for value sequence declaration
  template <typename BinaryOp> struct monoid_op;
  template <typename T> struct monoid_op<plus<T>> {
    static constexpr T e{0};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) + ...);
    }
  };
  template <typename T> struct monoid_op<multiplies<T>> {
    static constexpr T e{1};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) * ...);
    }
  };
  template <typename T> struct monoid_op<logical_or<T>> {
    static constexpr bool e{false};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) || ...);
    }
  };
  template <typename T> struct monoid_op<logical_and<T>> {
    static constexpr bool e{true};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) && ...);
    }
  };
  template <typename T> struct monoid_op<getmax<T>> {
    static constexpr T e{limits<T>::lowest()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res > args ? res : args), ...);
    }
  };
  template <typename T> struct monoid_op<getmin<T>> {
    static constexpr T e{limits<T>::max()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res < args ? res : args), ...);
    }
  };

  /// map operation
  struct count_leq {  ///< count less and equal
    template <typename... Tn> constexpr auto operator()(std::size_t M, Tn... Ns) const noexcept {
      if constexpr (sizeof...(Tn) > 0)
        return ((Ns <= M ? 1 : 0) + ...);
      else
        return 0;
    }
  };

}  // namespace zs
