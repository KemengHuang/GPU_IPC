#pragma once

#include <initializer_list>
#include <type_traits>

#include "../TypeAlias.hpp"

namespace zs {

  /// pre C++14 impl, https://zh.cppreference.com/w/cpp/types/void_t
  /// check ill-formed types
  // template <typename... Ts> struct make_void { using type = void; };
  // template <typename... Ts> using void_t = typename make_void<Ts...>::type;
  template <typename... Ts> using void_t = std::void_t<Ts...>;

  /// SFINAE
  template <bool B> struct enable_if;
  template <> struct enable_if<true> { using type = int; };
  template <bool B> using enable_if_t = typename enable_if<B>::type;
  template <bool... Bs> using enable_if_all = typename enable_if<(Bs && ...)>::type;
  template <bool... Bs> using enable_if_any = typename enable_if<(Bs || ...)>::type;
  /// underlying_type
  /// common_type

  ///
  /// type decorato
  ///
  template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_cvref_t = typename remove_cvref<T>::type;

  /// vref
  template <class T> struct remove_vref {
    using type = std::remove_volatile_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_vref_t = typename remove_vref<T>::type;

  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  /// decay+unref
  template <class T> struct unwrap_refwrapper { using type = T; };
  template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> { using type = T &; };
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename std::decay_t<T>>::type;

  ///
  /// fundamental type-value wrapper
  ///
  template <typename T> struct wrapt { using type = T; };
  /// wrap at most 1 layer
  template <typename T> struct wrapt<wrapt<T>> { using type = T; };
  template <typename T> constexpr wrapt<T> wrapt_v{};

  template <typename T> struct is_type_wrapper : std::false_type {};
  template <typename T> struct is_type_wrapper<wrapt<T>> : std::true_type {};
  template <typename T> constexpr bool is_type_wrapper_v = is_type_wrapper<T>::value;

  template <typename Tn, Tn N> using integral_t = std::integral_constant<Tn, N>;
  template <typename> struct is_integral : std::false_type {};
  template <typename T, T v> struct is_integral<integral_t<T, v>> : std::true_type {};

  template <auto N> using wrapv = integral_t<decltype(N), N>;
  template <auto N> constexpr wrapv<N> wrapv_v{};

  template <typename T> struct is_value_wrapper : std::false_type {};
  template <auto N> struct is_value_wrapper<wrapv<N>> : std::true_type {};
  template <typename T> constexpr bool is_value_wrapper_v = is_value_wrapper<T>::value;

  template <std::size_t N> using index_t = integral_t<std::size_t, N>;
  template <std::size_t N> constexpr index_t<N> index_c{};

#define INST_(T) std::declval<T>()
#define DECL_(T) std::declval<T>()

  constexpr std::true_type true_c{};
  constexpr std::false_type false_c{};

  /// arithmetic type
  constexpr wrapt<u8> u8_v{};
  constexpr wrapt<uint> uint_v{};
  constexpr wrapt<i16> i16_v{};
  constexpr wrapt<i32> i32_v{};
  constexpr wrapt<i64> i64_v{};
  constexpr wrapt<u16> u16_v{};
  constexpr wrapt<u32> u32_v{};
  constexpr wrapt<u64> u64_v{};
  constexpr wrapt<f32> f32_v{};
  constexpr wrapt<f64> f64_v{};
  template <typename T> constexpr wrapt<std::enable_if_t<std::is_arithmetic_v<T>, T>> number_v{};

  ///
  /// detection
  ///
  namespace detail {
    template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
    struct detector {
      using value_t = std::false_type;
      using type = Default;
    };

    template <class Default, template <class...> class Op, class... Args>
    struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
      using value_t = std::true_type;
      using type = Op<Args...>;
    };

  }  // namespace detail

  struct nonesuch {
    nonesuch() = delete;
    template <typename T> nonesuch(std::initializer_list<T>) = delete;
    ~nonesuch() = delete;
    nonesuch(const nonesuch &) = delete;
    void operator=(nonesuch const &) = delete;
  };

  template <template <class...> class Op, class... Args> using is_detected =
      typename detail::detector<nonesuch, void, Op, Args...>::value_t;

  template <template <class...> class Op, class... Args> using detected_t =
      typename detail::detector<nonesuch, void, Op, Args...>::type;

  template <typename Default, template <typename...> class Op, typename... Args> using detected_or
      = detail::detector<Default, void, Op, Args...>;

  template <typename Default, template <typename...> class Op, typename... Args> using detected_or_t
      = typename detected_or<Default, Op, Args...>::type;

  /// ref: boost-hana
  /// boost/hana/type.hpp
  // copyright Louis Dionne 2013-2017
  // Distributed under the Boost Software License, Version 1.0.
  // (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)
  //////////////////////////////////////////////////////////////////////////
  // is_valid
  //////////////////////////////////////////////////////////////////////////
  namespace detail {
    template <typename F, typename... Args> constexpr auto is_valid_impl(int) noexcept
        -> decltype(std::declval<F &&>()(std::declval<Args &&>()...), std::true_type{}) {
      return std::true_type{};
    }
    template <typename F, typename... Args> constexpr auto is_valid_impl(...) noexcept {
      return std::false_type{};
    }
    template <typename F> struct is_valid_fun {
      template <typename... Args> constexpr auto operator()(Args &&...) const noexcept {
        return is_valid_impl<F, Args &&...>(0);
      }
    };
  }  // namespace detail

  struct is_valid_t {
    template <typename F> constexpr auto operator()(F &&) const noexcept {
      return detail::is_valid_fun<F &&>{};
    }
    template <typename F, typename... Args> constexpr auto operator()(F &&, Args &&...) const {
      return detail::is_valid_impl<F &&, Args &&...>(0);
    }
  };

  constexpr is_valid_t is_valid{};

}  // namespace zs
