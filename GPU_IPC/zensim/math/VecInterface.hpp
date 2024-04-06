#pragma once
#include <tuple>

#include "MathUtils.h"
#include "../meta/Meta.h"
#include "../meta/Sequence.h"
#include "../types/Iterator.h"

namespace zs {

  template <typename, typename, typename> struct indexer_impl;

  /// indexer
  template <typename Tn, Tn... Ns, std::size_t... StorageOrders, std::size_t... Is>
  struct indexer_impl<integer_seq<Tn, Ns...>, index_seq<StorageOrders...>, index_seq<Is...>> {
    static_assert(sizeof...(Ns) == sizeof...(StorageOrders), "dimension mismatch");
    static_assert(std::is_integral_v<Tn>, "index type is not an integral");
    static constexpr int dim = sizeof...(Ns);
    using index_type = Tn;
    static constexpr index_type extent = (Ns * ...);
    using extents = integer_seq<Tn, Ns...>;
    using storage_orders = value_seq<StorageOrders...>;
    using lookup_orders = value_seq<storage_orders::template index<wrapv<Is>>::value...>;

    using storage_extents_impl
        = integer_seq<index_type, select_indexed_value<StorageOrders, Ns...>::value...>;
    using storage_bases = value_seq<excl_suffix_mul(Is, storage_extents_impl{})...>;
    using lookup_bases = value_seq<
        storage_bases::template type<lookup_orders::template type<Is>::value>::value...>;

    template <std::size_t I, enable_if_t<(I < dim)> = 0>
    static constexpr index_type storage_range() noexcept {
      return select_indexed_value<storage_orders::template type<I>::value, Ns...>::value;
    }
    template <std::size_t I, enable_if_t<(I < dim)> = 0>
    static constexpr index_type range() noexcept {
      return select_indexed_value<I, Ns...>::value;
    }

    template <std::size_t... Js, typename... Args>
    static constexpr index_type offset_impl(index_seq<Js...>, Args&&... args) noexcept {
      return (... + ((index_type)args * lookup_bases::template type<Js>::value));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    static constexpr index_type offset(Args&&... args) noexcept {
      return offset_impl(std::index_sequence_for<Args...>{}, FWD(args)...);
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == dim> = 0>
    static constexpr index_type offset(tuple<Args...> args) noexcept {
      return (... + ((index_type)get<Is>(args) * lookup_bases::template type<Is>::value));
    }
  };

  template <typename Tn, Tn... Ns> using indexer
      = indexer_impl<integer_seq<Tn, Ns...>, std::make_index_sequence<sizeof...(Ns)>,
                     std::make_index_sequence<sizeof...(Ns)>>;

  template <typename Orders, typename Tn, Tn... Ns> using ordered_indexer
      = indexer_impl<integer_seq<Tn, Ns...>, Orders, std::make_index_sequence<sizeof...(Ns)>>;

  template <typename Tensor, typename Extents> struct tensor_view;

  /**
   *    \note assume vec already defines base_t, value_type, index_type, extents and indexer_type
   *
   **/
#define SUPPLEMENT_VEC_STATIC_ATTRIBUTES                                                          \
  using dims = typename vseq_t<extents>::template to_iseq<sint_t>;                                \
  static constexpr index_type extent = vseq_t<extents>{}.reduce(multiplies<index_type>{});        \
  static constexpr int dim = vseq_t<extents>::count;                                              \
  template <std::size_t I, enable_if_t<(I < dim)> = 0> using range_t                              \
      = integral_t<index_type, select_value<I, vseq_t<extents>>::value>;                          \
  template <std::size_t I> static constexpr index_type range = range_t<I>::value;                 \
  using base_t::identity;                                                                         \
  using base_t::ones;                                                                             \
  using base_t::uniform;                                                                          \
  using base_t::zeros;                                                                            \
  template <typename OtherVecT,                                                                   \
            enable_if_all<OtherVecT::extent == extent,                                            \
                          std::is_convertible_v<typename OtherVecT::value_type, value_type>> = 0> \
  constexpr decltype(auto) operator=(const VecInterface<OtherVecT>& rhs) noexcept {               \
    for (index_type i = 0; i != extent; ++i) this->val(i) = rhs.val(i);                           \
    return *this;                                                                                 \
  }

  ///
  /// VecInterface
  ///
  // ref: https://en.cppreference.com/w/cpp/language/attributes/maybe_unused
  template <typename Derived> struct VecInterface {
#define DECLARE_VEC_INTERFACE_ATTRIBUTES                                           \
  using value_type [[maybe_unused]] = typename Derived::value_type;                \
  using index_type [[maybe_unused]] = typename Derived::index_type;                \
  using extents [[maybe_unused]] =                                                 \
      typename Derived::extents; /*not necessarily same as indexer_type::extents*/ \
  using indexer_type [[maybe_unused]] = typename Derived::indexer_type;            \
  using dims [[maybe_unused]] = typename Derived::dims;                            \
  enum : index_type { extent = Derived::extent };                                  \
  enum : int { dim = Derived::dim };

    using vec_type = Derived;

    constexpr auto derivedPtr() noexcept { return static_cast<Derived*>(this); }
    constexpr auto derivedPtr() const noexcept { return static_cast<const Derived*>(this); }
    constexpr auto derivedPtr() volatile noexcept { return static_cast<volatile Derived*>(this); }
    constexpr auto derivedPtr() const volatile noexcept {
      return static_cast<const volatile Derived*>(this);
    }

    template <typename VecT = Derived> static constexpr bool is_access_lref
        = std::is_lvalue_reference_v<decltype(std::declval<VecT&>().val(0))>;
    template <typename VecT = Derived> using variant_vec_type =
        typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents>;
    template <typename VecT = Derived> static constexpr bool is_variant_writable
        = std::is_lvalue_reference_v<decltype(std::declval<variant_vec_type<VecT>&>().val(0))>;

    constexpr auto data() noexcept { return derivedPtr()->do_data(); }
    constexpr auto data() volatile noexcept {
      return static_cast<volatile Derived*>(this)->do_data();
    }
    constexpr auto data() const noexcept { return derivedPtr()->do_data(); }
    constexpr auto data() const volatile noexcept {
      return static_cast<const volatile Derived*>(this)->do_data();
    }
    /// property query
    static constexpr auto get_extent() noexcept { return Derived::extent; }
    static constexpr auto get_dim() noexcept { return Derived::dim; }
    static constexpr auto get_dims() noexcept { return wrapt<typename Derived::dims>{}; }

    struct detail {
      template <typename VecT, std::size_t... Is>
      static constexpr bool all_the_same_dimension_extent(typename VecT::index_type v,
                                                          index_seq<Is...>) noexcept {
        return ((VecT::template range<Is> == v) && ...);
      }
    };

    template <typename VecT> static constexpr bool same_extent_each_dimension() noexcept {
      return detail::template all_the_same_dimension_extent<VecT>(
          VecT::template range_t<0>::value, std::make_index_sequence<VecT::dim>{});
    }

    ///
    /// entry access
    ///
    // ()
    template <typename VecT = Derived, typename... Tis,
              enable_if_all<sizeof...(Tis) <= VecT::dim, (std::is_integral_v<Tis> && ...)> = 0>
    constexpr decltype(auto) operator()(Tis... is) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return derivedPtr()->operator()(is...);
    }
    template <typename VecT = Derived, typename... Tis,
              enable_if_all<sizeof...(Tis) <= VecT::dim, (std::is_integral_v<Tis> && ...)> = 0>
    constexpr decltype(auto) operator()(Tis... is) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return derivedPtr()->operator()(is...);
    }
    // val (one dimension index)
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) val(Ti i) noexcept {
      return derivedPtr()->do_val(i);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) val(Ti i) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return derivedPtr()->do_val(i);
    }
    // [] (one dimension index)
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) noexcept {
      return derivedPtr()->operator[](is);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) const noexcept {
      return derivedPtr()->operator[](is);
    }
    // tuple as index
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<sizeof...(Ts) <= VecT::dim,
                            (std::is_integral_v<remove_cvref_t<Ts>> && ...)> = 0>
    constexpr decltype(auto) val(const std::tuple<Ts...>& is) noexcept {
      return std::apply(static_cast<Derived&>(*this), is);
    }
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<(sizeof...(Ts) <= VecT::dim),
                            (std::is_integral_v<remove_cvref_t<Ts>> && ...)> = 0>
    constexpr decltype(auto) val(const std::tuple<Ts...>& is) const noexcept {
      return std::apply(static_cast<const Derived&>(*this), is);
    }
    // tuple as index
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<sizeof...(Ts) <= VecT::dim,
                            (std::is_integral_v<remove_cvref_t<Ts>> && ...)> = 0>
    constexpr decltype(auto) val(const zs::tuple<Ts...>& is) noexcept {
      return zs::apply(static_cast<Derived&>(*this), is);
    }
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<(sizeof...(Ts) <= VecT::dim),
                            (std::is_integral_v<remove_cvref_t<Ts>> && ...)> = 0>
    constexpr decltype(auto) val(const zs::tuple<Ts...>& is) const noexcept {
      return zs::apply(static_cast<const Derived&>(*this), is);
    }

    ///
    /// construction
    ///
    template <typename T, typename VecT = Derived,
              enable_if_all<std::is_convertible_v<T, typename VecT::value_type>> = 0>
    static constexpr auto uniform(const T& v) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = v;
      return r;
    }
    template <typename F, typename VecT = Derived,
              enable_if_all<std::is_convertible_v<
                  decltype(std::declval<F>()(std::declval<typename VecT::index_type>())),
                  typename VecT::value_type>> = 0>
    static constexpr auto init(const F& f) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = f(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_all<std::is_convertible_v<int, typename VecT::value_type>> = 0>
    static constexpr auto zeros() noexcept {
      return uniform(0);
    }
    template <typename VecT = Derived,
              enable_if_all<std::is_convertible_v<int, typename VecT::value_type>> = 0>
    static constexpr auto ones() noexcept {
      return uniform(1);
    }
    template <typename VecT = Derived, enable_if_all<same_extent_each_dimension<VecT>()> = 0>
    static constexpr auto identity() noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      auto r = zeros();
      constexpr index_type N = VecT::template range_t<0>::value;
      for (index_type i = 0; i != N; ++i)
        r.val(gen_seq<VecT::dim>::template uniform_values<std::tuple>(i)) = 1;
      return r;
    }
    template <typename VecT = Derived> constexpr auto vectorize() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, integer_seq<index_type, Derived::extent>>
          r{};
      for (index_type i = 0; i != Derived::extent; ++i) r.val(i) = derivedPtr()->val(i);
      return r;
    }
    constexpr auto clone() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = derivedPtr()->val(i);
      return r;
    }
    template <typename T, typename VecT = Derived,
              enable_if_t<std::is_convertible_v<typename VecT::value_type, T>> = 0>
    constexpr auto cast() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<T, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = static_cast<T>(derivedPtr()->val(i));
      return r;
    }
    template <typename T, typename VecT = Derived,
              enable_if_t<sizeof(typename VecT::value_type) == sizeof(T)> = 0>
    constexpr auto reinterpret_bits() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<T, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = zs::reinterpret_bits<T>(derivedPtr()->val(i));
      return r;
    }
    template <typename T, typename VecT = Derived,
              enable_if_t<std::is_convertible_v<typename VecT::value_type, T>> = 0>
    constexpr operator typename VecT::template variant_vec<T, typename VecT::extents>()
        const noexcept {
      return derivedPtr()->template cast<T>();
    }

    template <
        typename T, typename VecT = Derived,
        enable_if_all<std::is_convertible_v<remove_cvref_t<T>, typename VecT::value_type>> = 0>
    constexpr Derived& operator=(T&& v) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i) derivedPtr()->val(i) = v;
      return static_cast<Derived&>(*this);
    }
    template <typename T, typename VecT = Derived,
              enable_if_all<std::is_convertible_v<T, typename VecT::value_type>> = 0>
    constexpr Derived& operator=(const std::initializer_list<T>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      index_type i = 0;
      for (auto&& v : rhs) {
        derivedPtr()->val(i++) = v;
        if (i == extent) break;
      }
      return static_cast<Derived&>(*this);
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<OtherVecT::extent == VecT::extent,
                            // otherwise triggers copy-assign ctor
                            !is_same_v<OtherVecT, VecT>,
                            std::is_convertible_v<typename OtherVecT::value_type,
                                                  typename VecT::value_type>> = 0>
    constexpr Derived& operator=(const VecInterface<OtherVecT>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i) derivedPtr()->val(i) = rhs.val(i);
      return static_cast<Derived&>(*this);
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<OtherVecT::extent == VecT::extent,
                            std::is_convertible_v<typename OtherVecT::value_type,
                                                  typename VecT::value_type>> = 0>
    constexpr Derived& assign(const VecInterface<OtherVecT>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i) derivedPtr()->val(i) = rhs.val(i);
      return static_cast<Derived&>(*this);
    }
    template <typename T, typename VecT = Derived,
              enable_if_all<std::is_convertible_v<T, typename VecT::value_type>> = 0>
    constexpr Derived& operator=(const std::array<T, VecT::extent>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i) derivedPtr()->val(i) = rhs[i];
      return static_cast<Derived&>(*this);
    }
    constexpr operator Derived&() noexcept { return *derivedPtr(); }
    constexpr operator const Derived&() const noexcept { return *derivedPtr(); }

    ///
    /// arithmetic operators
    /// ref: https://github.com/cemyuksel/cyCodeBase/blob/master/cyIVector.h
    ///
    //!@name Unary operators
    constexpr auto operator-() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = -derivedPtr()->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_integral_v<typename VecT::value_Type>> = 0>
    constexpr auto operator!() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<bool, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = !static_cast<bool>(derivedPtr()->val(i));
      return r;
    }
    template <typename VecT = Derived, enable_if_t<VecT::dim == 2> = 0>
    constexpr auto transpose() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      constexpr auto M = select_value<0, vseq_t<extents>>::value;
      constexpr auto N = select_value<1, vseq_t<extents>>::value;
      typename Derived::template variant_vec<value_type, integer_seq<index_type, N, M>> r{};
      for (index_type i = 0; i != M; ++i)
        for (index_type j = 0; j != N; ++j) r(j, i) = (*this)(i, j);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto prod() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{1};
      for (index_type i = 0; i != extent; ++i) res *= derivedPtr()->val(i);
      return res;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto sum() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i) res += derivedPtr()->val(i);
      return res;
      // return res.sum();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto l2NormSqr() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i) res += derivedPtr()->val(i) * derivedPtr()->val(i);
      return res;
      // return res.sum();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto infNormSqr() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i)
        if (value_type sqr = derivedPtr()->val(i) * derivedPtr()->val(i); sqr > res) res = sqr;
      return res;
    }

    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto length() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                              conditional_t<(sizeof(value_type) >= 8), double, float>>;
      return zs::sqrt((T)l2NormSqr());
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto norm() const noexcept {
      return length();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto infNorm() const noexcept {
      return abs().max();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto normalized() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      const auto len = length();
      typename Derived::template variant_vec<RM_CVREF_T(len), extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = derivedPtr()->val(i) / len;
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto max() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{derivedPtr()->val(0)};
      for (index_type i = 1; i != extent; ++i)
        if (const auto& v = derivedPtr()->val(i); v > res) res = v;
      return res;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto max(typename VecT::value_type val) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        if (const auto& v = derivedPtr()->val(i); v > val)
          r.val(i) = v;
        else
          r.val(i) = val;
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto min() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{derivedPtr()->val(0)};
      for (index_type i = 1; i != extent; ++i)
        if (const auto& v = derivedPtr()->val(i); v < res) res = v;
      return res;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto min(typename VecT::value_type val) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        if (const auto& v = derivedPtr()->val(i); v < val)
          r.val(i) = v;
        else
          r.val(i) = val;
      return r;
    }
    template <typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>, VecT::dim == 1,
                            VecT::extent == 2> = 0>
    constexpr auto orthogonal() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                              conditional_t<(sizeof(value_type) >= 8), f64, f32>>;
      using V = typename Derived::template variant_vec<T, extents>;
      V r{};
      r.val(0) = -derivedPtr()->val(1);
      r.val(1) = derivedPtr()->val(0);
      return r.normalized();
    }
    template <typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>, VecT::dim == 1,
                            VecT::extent == 3> = 0>
    constexpr auto orthogonal() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                              conditional_t<(sizeof(value_type) >= 8), f64, f32>>;
      T x = math::abs(derivedPtr()->val(0));
      T y = math::abs(derivedPtr()->val(1));
      T z = math::abs(derivedPtr()->val(2));
      using V = typename Derived::template variant_vec<T, extents>;
      auto r = V::zeros();
      if (x < y) {
        if (x < z)
          r.val(0) = 1;
        else
          r.val(2) = 1;
      } else {
        if (y < z)
          r.val(1) = 1;
        else
          r.val(2) = 1;
      }
      // auto other = x < y ? (x < z ? V{1, 0, 0} : V{0, 0, 1}) : (y < z ? V{0, 1, 0} : V{0, 0, 1});
      return cross(r);
    }
    //!@name Coefficient-wise math funcs
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto abs() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = derivedPtr()->val(i) > 0 ? derivedPtr()->val(i) : -derivedPtr()->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto reciprocal() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = math::near_zero(derivedPtr()->val(i)) ? limits<value_type>::infinity()
                                                         : (value_type)1 / derivedPtr()->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto exp() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = zs::exp(derivedPtr()->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto log() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = zs::log(derivedPtr()->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto log1p() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = zs::log1p(derivedPtr()->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = -1>
    constexpr auto sqrt() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = zs::sqrt(derivedPtr()->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto square() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = derivedPtr()->val(i) * derivedPtr()->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto cube() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = derivedPtr()->val(i) * derivedPtr()->val(i) * derivedPtr()->val(i);
      return r;
    }
    template <
        typename VecT = Derived,
        enable_if_all<std::is_floating_point_v<typename VecT::value_type>, VecT::dim == 1> = 0>
    constexpr auto deviatoric() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      auto rightTerm = -sum() / (value_type)extent;
      for (index_type i = 0; i != extent; ++i) r.val(i) = derivedPtr()->val(i) + rightTerm;
      return r;
    }

    //!@name Binary operators
    // scalar
#define DEFINE_VEC_OP_SCALAR(OP)                                                              \
  template <typename TT, typename VecT = Derived,                                             \
            enable_if_t<std::is_convertible_v<                                                \
                            TT, typename VecT::value_type> && std::is_fundamental_v<TT>> = 0> \
  friend constexpr auto operator OP(VecInterface const& e, TT const v) noexcept {             \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                          \
    using R = math::op_result_t<value_type, TT>;                                              \
    typename Derived::template variant_vec<R, extents> r{};                                   \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)e.val(i) OP((R)v);                 \
    return r;                                                                                 \
  }                                                                                           \
  template <typename TT, typename VecT = Derived,                                             \
            enable_if_t<std::is_convertible_v<                                                \
                            TT, typename VecT::value_type> && std::is_fundamental_v<TT>> = 0> \
  friend constexpr auto operator OP(TT const v, VecInterface const& e) noexcept {             \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                          \
    using R = math::op_result_t<value_type, TT>;                                              \
    typename Derived::template variant_vec<R, extents> r{};                                   \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)v OP((R)e.val(i));                 \
    return r;                                                                                 \
  }
    DEFINE_VEC_OP_SCALAR(+)
    DEFINE_VEC_OP_SCALAR(-)
    DEFINE_VEC_OP_SCALAR(*)
    DEFINE_VEC_OP_SCALAR(/)

// scalar integral
#define DEFINE_VEC_OP_SCALAR_INTEGRAL(OP)                                                       \
  template <                                                                                    \
      typename TT, typename VecT = Derived,                                                     \
      enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>> = 0> \
  friend constexpr auto operator OP(VecInterface const& e, TT const v) noexcept {               \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                            \
    using R = math::op_result_t<value_type, TT>;                                                \
    typename Derived::template variant_vec<R, extents> r{};                                     \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)e.val(i) OP((R)v);                   \
    return r;                                                                                   \
  }                                                                                             \
  template <                                                                                    \
      typename TT, typename VecT = Derived,                                                     \
      enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>> = 0> \
  friend constexpr auto operator OP(TT const v, VecInterface const& e) noexcept {               \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                            \
    using R = math::op_result_t<value_type, TT>;                                                \
    typename Derived::template variant_vec<R, extents> r{};                                     \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)v OP((R)e.val(i));                   \
    return r;                                                                                   \
  }
    DEFINE_VEC_OP_SCALAR_INTEGRAL(&)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(|)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(^)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(>>)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(<<)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(%)

    // vector
#define DEFINE_VEC_OP_VECTOR(OP)                                                         \
  template <typename OtherVecT, typename VecT = Derived,                                 \
            enable_if_t<is_same_v<typename OtherVecT::dims, typename VecT::dims>> = 0>   \
  friend constexpr auto operator OP(VecInterface const& lhs,                             \
                                    VecInterface<OtherVecT> const& rhs) noexcept {       \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                     \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;             \
    typename Derived::template variant_vec<R, extents> r{};                              \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) OP((R)rhs.val(i)); \
    return r;                                                                            \
  }
    DEFINE_VEC_OP_VECTOR(+)
    DEFINE_VEC_OP_VECTOR(-)
    DEFINE_VEC_OP_VECTOR(/)

    template <typename VecTA, typename VecTB>
    static constexpr bool is_matrix_matrix_product() noexcept {
      if constexpr (VecTA::dim == 2 && VecTB::dim == 2) {
        if constexpr (VecTA::template range_t<1>::value == VecTB::template range_t<0>::value)
          return true;
        else
          return false;
      } else
        return false;
    }
    /// coeffwise product
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<is_same_v<typename OtherVecT::dims, typename VecT::dims>,
                            !is_matrix_matrix_product<VecT, OtherVecT>()> = 0>
    friend constexpr auto operator*(const VecInterface& lhs,
                                    const VecInterface<OtherVecT>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      typename Derived::template variant_vec<R, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) * ((R)rhs.val(i));
      return r;
    }
    /// matrix-matrix product
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<is_matrix_matrix_product<VecT, OtherVecT>()> = 0>
    friend constexpr auto operator*(const VecInterface& lhs,
                                    const VecInterface<OtherVecT>& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      constexpr auto Ni = VecT::template range_t<0>::value;
      constexpr auto Nj = OtherVecT::template range_t<1>::value;
      constexpr auto Nk = VecT::template range_t<1>::value;
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      typename Derived::template variant_vec<R, integer_seq<index_type, Ni, Nj>> r{};
      for (index_type i = 0; i != Ni; ++i)
        for (index_type j = 0; j != Nj; ++j) {
          r(i, j) = 0;
          for (index_type k = 0; k != Nk; ++k) r(i, j) += lhs(i, k) * rhs(k, j);
        }
      return r;
    }
    /// matrix-vector product
    template <typename VecTV, typename VecTM = Derived,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<1>::value == VecTV::extent> = 0>
    friend constexpr auto operator*(const VecInterface& A, const VecInterface<VecTV>& x) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      constexpr auto M = VecTM::template range_t<0>::value;
      constexpr auto N = VecTM::template range_t<1>::value;
      using R = math::op_result_t<value_type, typename VecTV::value_type>;
      typename Derived::template variant_vec<R, integer_seq<index_type, M>> r{};
      for (index_type i = 0; i != M; ++i) {
        r(i) = 0;
        for (index_type j = 0; j != N; ++j) r(i) += A(i, j) * x(j);
      }
      return r;
    }
    template <typename VecTM, typename VecTV = Derived,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTV::extent> = 0>
    friend constexpr auto operator*(const VecInterface& x, const VecInterface<VecTM>& A) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      constexpr auto M = VecTM::template range_t<0>::value;
      constexpr auto N = VecTM::template range_t<1>::value;
      using R = math::op_result_t<value_type, typename VecTM::value_type>;
      typename Derived::template variant_vec<R, integer_seq<index_type, N>> r{};
      for (index_type j = 0; j != N; ++j) {
        r(j) = 0;
        for (index_type i = 0; i != M; ++i) r(j) += A(i, j) * x(i);
      }
      return r;
    }
    // DEFINE_VEC_OP_VECTOR_GENERAL(*)

    // vector integral
#define DEFINE_VEC_OP_VECTOR_INTEGRAL(OP)                                                \
  template <typename OtherVecT, typename VecT = Derived,                                 \
            enable_if_all<is_same_v<typename OtherVecT::dims, typename VecT::dims>,      \
                          std::is_integral_v<typename OtherVecT::value_type>,            \
                          std::is_integral_v<typename VecT::value_type>> = 0>            \
  friend constexpr auto operator OP(VecInterface const& lhs,                             \
                                    VecInterface<OtherVecT> const& rhs) noexcept {       \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                     \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;             \
    typename Derived::template variant_vec<R, extents> r{};                              \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) OP((R)rhs.val(i)); \
    return r;                                                                            \
  }
    DEFINE_VEC_OP_VECTOR_INTEGRAL(&)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(|)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(^)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(>>)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(<<)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(%)

//!@name Assignment operators
// scalar
#define DEFINE_VEC_OP_SCALAR_ASSIGN(OP)                                                            \
  template <typename TT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>,        \
            enable_if_all<std::is_convertible_v<TT, typename VecT::value_type>, IsAssignable> = 0> \
  constexpr Derived& operator OP##=(TT v) noexcept {                                               \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                               \
    using R = math::op_result_t<value_type, TT>;                                                   \
    for (index_type i = 0; i != extent; ++i)                                                       \
      derivedPtr()->val(i) = (R)derivedPtr()->val(i) OP((R)v);                                     \
    return static_cast<Derived&>(*this);                                                           \
  }
    DEFINE_VEC_OP_SCALAR_ASSIGN(+)
    DEFINE_VEC_OP_SCALAR_ASSIGN(-)
    DEFINE_VEC_OP_SCALAR_ASSIGN(*)
    DEFINE_VEC_OP_SCALAR_ASSIGN(/)

    // scalar integral
#define DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(OP)                                                 \
  template <typename TT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>,      \
            enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>, \
                          IsAssignable> = 0>                                                     \
  constexpr Derived& operator OP##=(TT v) noexcept {                                             \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                             \
    using R = math::op_result_t<value_type, TT>;                                                 \
    for (index_type i = 0; i != extent; ++i)                                                     \
      derivedPtr()->val(i) = (R)derivedPtr()->val(i) OP((R)v);                                   \
    return static_cast<Derived&>(*this);                                                         \
  }
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(&)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(|)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(^)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(>>)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(<<)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(%)

    // vector
#define DEFINE_VEC_OP_VECTOR_ASSIGN(OP)                                                            \
  template <typename OtherVecT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>, \
            enable_if_all<                                                                         \
                std::is_convertible_v<typename OtherVecT::value_type, typename VecT::value_type>,  \
                IsAssignable> = 0>                                                                 \
  constexpr Derived& operator OP##=(VecInterface<OtherVecT> const& o) noexcept {                   \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                               \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                       \
    for (index_type i = 0; i != extent; ++i)                                                       \
      derivedPtr()->val(i) = (R)derivedPtr()->val(i) OP((R)o.val(i));                              \
    return static_cast<Derived&>(*this);                                                           \
  }
    DEFINE_VEC_OP_VECTOR_ASSIGN(+)
    DEFINE_VEC_OP_VECTOR_ASSIGN(-)
    DEFINE_VEC_OP_VECTOR_ASSIGN(*)
    DEFINE_VEC_OP_VECTOR_ASSIGN(/)

#define DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(OP)                                                   \
  template <typename OtherVecT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>, \
            enable_if_all<std::is_integral_v<typename VecT::value_type>,                           \
                          std::is_integral_v<typename OtherVecT::value_type>, IsAssignable> = 0>   \
  constexpr Derived& operator OP##=(VecInterface<OtherVecT> const& o) noexcept {                   \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                               \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                       \
    for (index_type i = 0; i != extent; ++i)                                                       \
      derivedPtr()->val(i) = (R)derivedPtr()->val(i) OP((R)o.val(i));                              \
    return static_cast<Derived&>(*this);                                                           \
  }
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(&)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(|)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(^)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(>>)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(<<)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(%)

    ///
    /// linalg
    ///
    // member func version
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>> = 0>
    constexpr auto dot(VecInterface<OtherVecT> const& rhs) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      R res{0};
      for (index_type i = 0; i != extent; ++i) res += derivedPtr()->val(i) * rhs.val(i);
      return res;
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == 2, OtherVecT::extent == 2> = 0>
    constexpr auto cross(VecInterface<OtherVecT> const& rhs) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      return (R)derivedPtr()->val(0) * (R)rhs.val(1) - (R)derivedPtr()->val(1) * (R)rhs.val(0);
    }

    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == 3, OtherVecT::extent == 3> = 0>
    constexpr auto cross(VecInterface<OtherVecT> const& rhs) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      typename Derived::template variant_vec<R, extents> res{};  // extents type follows 'lhs'
      res.val(0)
          = (R)derivedPtr()->val(1) * (R)rhs.val(2) - (R)derivedPtr()->val(2) * (R)rhs.val(1);
      res.val(1)
          = (R)derivedPtr()->val(2) * (R)rhs.val(0) - (R)derivedPtr()->val(0) * (R)rhs.val(2);
      res.val(2)
          = (R)derivedPtr()->val(0) * (R)rhs.val(1) - (R)derivedPtr()->val(1) * (R)rhs.val(0);
      return res;
    }
    // friend version
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>> = 0>
    friend constexpr auto dot(VecInterface const& lhs,
                              VecInterface<OtherVecT> const& rhs) noexcept {
      return lhs.dot(rhs);
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == OtherVecT::extent,
                            (VecT::extent >= 2), (VecT::extent <= 3)> = 0>
    friend constexpr auto cross(VecInterface const& lhs,
                                VecInterface<OtherVecT> const& rhs) noexcept {
      return lhs.cross(rhs);
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == OtherVecT::extent,
                            (VecT::extent >= 2), (VecT::extent <= 3)> = 0>
    friend constexpr bool parallel(VecInterface const& lhs,
                                   VecInterface<OtherVecT> const& rhs) noexcept {
      constexpr auto dim = Derived::dim;
      using index_type = typename Derived::index_type;
      const auto c = lhs.cross(rhs);
      for (index_type d = 0; d != dim; ++d)
        if (math::near_zero(c[d])) return true;
      return false;
    }

    ///
    /// compare
    ///
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_t<is_same_v<typename OtherVecT::dims, typename VecT::dims>> = 0>
    friend constexpr bool operator==(VecInterface const& lhs,
                                     VecInterface<OtherVecT> const& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i)
        if (lhs.val(i) != rhs.val(i)) return false;
      return true;
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_t<is_same_v<typename OtherVecT::dims, typename VecT::dims>> = 0>
    friend constexpr bool operator!=(VecInterface const& lhs,
                                     VecInterface<OtherVecT> const& rhs) noexcept {
      return !(lhs == rhs);
    }

  protected:
    constexpr auto do_data() noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (value_type*)nullptr;
    }
    constexpr auto do_data() volatile noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (volatile value_type*)nullptr;
    }
    constexpr auto do_data() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (const value_type*)nullptr;
    }
    constexpr auto do_data() const volatile noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (const volatile value_type*)nullptr;
    }
  };

  namespace detail {
    template <typename VecT, std::size_t dim, sint_t... dims, std::size_t... Is>
    constexpr bool vec_fits_shape(integer_seq<sint_t, dims...>, index_seq<Is...>) noexcept {
      static_assert(sizeof...(dims) == sizeof...(Is), "count of indices and dims mismatch.");
      if constexpr (dim == VecT::dim) {
        if constexpr (sizeof...(dims) <= dim)
          return ((VecT::template range_t<Is>::value == select_indexed_value<Is, dims...>::value)
                  && ...);
        else
          return false;
      } else
        return false;
    }
  }  // namespace detail

  template <typename VecT, std::size_t dim, sint_t... dims>
  constexpr bool vec_fits_shape() noexcept {
    return detail::vec_fits_shape<VecT, dim>(integer_seq<sint_t, dims...>{},
                                             std::make_index_sequence<sizeof...(dims)>{});
  }

  template <typename T, typename = void> struct is_vec : std::false_type {};
  template <typename VecT> struct is_vec<VecInterface<VecT>> : std::true_type {};
  template <typename VecT>
  struct is_vec<VecT, std::enable_if_t<std::is_base_of_v<VecInterface<VecT>, VecT>>>
      : std::true_type {};

  template <std::size_t d = 0, typename VecTV, typename VecTM,
            enable_if_all<VecTV::dim == 1, VecTM::dim + d == VecTV::extent> = 0>
  constexpr auto xlerp(const VecInterface<VecTV>& diff, const VecInterface<VecTM>& arena) noexcept {
    if constexpr (VecTM::dim > 1)
      return linear_interop(diff(d), xlerp<d + 1>(diff, arena[0]), xlerp<d + 1>(diff, arena[1]));
    else
      return linear_interop(diff(d), arena(0), arena(1));
  }

}  // namespace zs