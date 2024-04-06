#pragma once

#include <utility>

#include "../Reflection.h"
#include "../math/MathUtils.h"
#include "../meta/ControlFlow.h"
#include "../meta/Meta.h"
#include "../meta/Relationship.h"
#include "../meta/Sequence.h"

namespace zs {

  template <typename T> struct wrapt;
  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;
  template <typename... Seqs> struct concat;
  template <typename> struct VecInterface;

  template <std::size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <std::size_t I, typename... Ts> using select_indexed_type
      = select_type<I, type_seq<Ts...>>;

  template <class T> struct unwrap_refwrapper;
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename std::decay_t<T>>::type;

/// Jorg Brown, Cppcon2019, reducing template compilation overhead using
/// features from C++11, 14, 17, and 20
#if 0
template <std::size_t I, typename T> struct tuple_value {
  /// by index
  constexpr T &get(std::integral_constant<std::size_t, I>) noexcept {
    return value;
  }
  constexpr T const &
  get(std::integral_constant<std::size_t, I>) const noexcept {
    return value;
  }
  /// by type
  constexpr T &get(wrapt<T>) noexcept { return value; }
  constexpr T const &get(wrapt<T>) const noexcept { return value; }
  T value;
};
#else
  template <std::size_t I, typename T, typename = void> struct tuple_value : T {
    constexpr tuple_value() = default;
    ~tuple_value() = default;
    template <typename V> constexpr tuple_value(V &&v) noexcept : T{FWD(v)} {}
    constexpr tuple_value(tuple_value &&) = default;
    constexpr tuple_value(const tuple_value &) = default;
    constexpr tuple_value &operator=(tuple_value &&) = default;
    constexpr tuple_value &operator=(const tuple_value &) = default;

    /// by index
    constexpr decltype(auto) get(integral_t<std::size_t, I>) &noexcept { return *this; }
    constexpr decltype(auto) get(integral_t<std::size_t, I>) &&noexcept { return std::move(*this); }
    constexpr decltype(auto) get(integral_t<std::size_t, I>) const &noexcept { return *this; }
    /// by type
    constexpr decltype(auto) get(wrapt<T>) &noexcept { return *this; }
    constexpr decltype(auto) get(wrapt<T>) &&noexcept { return std::move(*this); }
    constexpr decltype(auto) get(wrapt<T>) const &noexcept { return *this; }
  };
  template <std::size_t I, typename T> struct tuple_value<
      I, T,
      std::enable_if_t<(
          std::is_fundamental_v<
              T> || std::is_final_v<T> || std::is_same_v<T, void *> || std::is_reference_v<T> || std::is_pointer_v<T>)>> {
    constexpr tuple_value() = default;
    ~tuple_value() = default;
    template <typename V> constexpr tuple_value(V &&v) noexcept : value{FWD(v)} {}
    constexpr tuple_value(tuple_value &&) = default;
    constexpr tuple_value(const tuple_value &) = default;
    constexpr tuple_value &operator=(tuple_value &&) = default;
    constexpr tuple_value &operator=(const tuple_value &) = default;

    /// by index
    constexpr decltype(auto) get(integral_t<std::size_t, I>) &noexcept {
      if constexpr (std::is_rvalue_reference_v<T>)
        return std::move(value);
      else
        return value;
    }
    constexpr decltype(auto) get(integral_t<std::size_t, I>) &&noexcept { return std::move(value); }
    template <bool NonRValRef = !std::is_rvalue_reference_v<T>, enable_if_t<NonRValRef> = 0>
    constexpr decltype(auto) get(integral_t<std::size_t, I>) const &noexcept {
      return value;
    }
    /// by type
    constexpr decltype(auto) get(wrapt<T>) &noexcept {
      if constexpr (std::is_rvalue_reference_v<T>)
        return std::move(value);
      else
        return value;
    }
    constexpr decltype(auto) get(wrapt<T>) &&noexcept { return std::move(value); }
    template <bool NonRValRef = !std::is_rvalue_reference_v<T>, enable_if_t<NonRValRef> = 0>
    constexpr decltype(auto) get(wrapt<T>) const &noexcept {
      return value;
    }
    T value;
  };
#endif

  template <typename, typename> struct tuple_base;
  template <typename... Ts> struct tuple;

  template <typename> struct is_tuple : std::false_type {};
  template <typename... Ts> struct is_tuple<tuple<Ts...>> : std::true_type {};
  template <typename T> static constexpr bool is_tuple_v = is_tuple<T>::value;

  template <typename... Args> constexpr auto make_tuple(Args &&...args);
  template <typename T> struct tuple_size;
  template <typename... Ts> struct tuple_size<tuple<Ts...>>
      : std::integral_constant<std::size_t, sizeof...(Ts)> {};
  template <typename Tup> constexpr std::enable_if_t<is_tuple_v<Tup>, std::size_t> tuple_size_v
      = tuple_size<Tup>::value;

  template <std::size_t... Is, typename... Ts> struct tuple_base<index_seq<Is...>, type_seq<Ts...>>
      : tuple_value<Is, Ts>... {
    using tuple_types = type_seq<Ts...>;
    static constexpr std::size_t tuple_size = sizeof...(Ts);

    constexpr tuple_base() = default;
    ~tuple_base() = default;
    template <typename... Vs, enable_if_t<sizeof...(Vs) == tuple_size> = 0>
    constexpr tuple_base(Vs &&...vs) noexcept : tuple_value<Is, Ts>{FWD(vs)}... {}
    constexpr tuple_base(tuple_base &&) = default;
    constexpr tuple_base(const tuple_base &) = default;
    constexpr tuple_base &operator=(tuple_base &&) = default;
    constexpr tuple_base &operator=(const tuple_base &) = default;

    using tuple_value<Is, Ts>::get...;
    template <std::size_t I> constexpr decltype(auto) get() noexcept {
      return get(integral_t<std::size_t, I>{});
    }
    template <std::size_t I> constexpr decltype(auto) get() const noexcept {
      return get(integral_t<std::size_t, I>{});
    }
    template <typename T> constexpr decltype(auto) get() noexcept { return get(wrapt<T>{}); }
    template <typename T> constexpr decltype(auto) get() const noexcept { return get(wrapt<T>{}); }
    /// custom
    constexpr auto &head() noexcept { return get(integral_t<std::size_t, 0>{}); }
    constexpr auto const &head() const noexcept { return get(integral_t<std::size_t, 0>{}); }
    constexpr auto &tail() noexcept { return get(integral_t<std::size_t, tuple_size - 1>{}); }
    constexpr auto const &tail() const noexcept {
      return get(integral_t<std::size_t, tuple_size - 1>{});
    }
    constexpr decltype(auto) std() const noexcept { return std::forward_as_tuple(get<Is>()...); }
    constexpr decltype(auto) std() noexcept { return std::forward_as_tuple(get<Is>()...); }
    /// iterator
    /// compwise
    template <typename BinaryOp, typename... TTs>
    constexpr auto compwise(BinaryOp &&op, const tuple<TTs...> &t) const noexcept {
      return zs::make_tuple(op(get<Is>(), t.template get<Is>())...);
    }
    template <typename BinaryOp, auto... Ns>
    constexpr auto compwise(BinaryOp &&op, value_seq<Ns...>) const noexcept {
      return zs::make_tuple(op(get<Is>(), Ns)...);
    }
    /// for_each
    template <typename UnaryOp> constexpr auto for_each(UnaryOp &&op) const noexcept {
      // https://en.cppreference.com/w/cpp/language/eval_order
      // In the evaluation of each of the following four expressions, using
      // the built-in (non-overloaded) operators, there is a sequence point
      // after the evaluation of the expression a. a && b a || b a ? b : c a ,
      // b
      return (op(get<Is>()), ...);
    }
    /// map
    template <typename MapOp, std::size_t... Js>
    constexpr auto map_impl(MapOp &&op, std::index_sequence<Js...>) const noexcept {
      return zs::make_tuple(op(Js, get<Is>()...)...);
    }
    template <std::size_t N, typename MapOp> constexpr auto map(MapOp &&op) const noexcept {
      return map_impl(std::forward<MapOp>(op), gen_seq<N>::ascend());
    }
    /// reduce
    template <typename MonoidOp> constexpr auto reduce(MonoidOp &&op) const noexcept {
      return op(get<Is>()...);
    }
    template <typename UnaryOp, typename MonoidOp>
    constexpr auto reduce(UnaryOp &&uop, MonoidOp &&mop) const noexcept {
      return mop(uop(get<Is>())...);
    }
    template <typename BinaryOp, typename MonoidOp, auto... Ns>
    constexpr auto reduce(BinaryOp &&bop, MonoidOp &&mop, value_seq<Ns...>) const noexcept {
      return mop(bop(get<Is>(), Ns)...);
    }
    template <typename BinaryOp, typename MonoidOp, typename... TTs> constexpr auto reduce(
        BinaryOp &&bop, MonoidOp &&mop,
        const tuple_base<std::index_sequence<Is...>, type_seq<TTs...>> &t) const noexcept {
      return mop(bop(get<Is>(), t.template get<Is>())...);
    }
    /// shuffle
    template <typename... Args> constexpr auto shuffle(Args &&...args) const noexcept {
      return zs::make_tuple(get<FWD(args)>()...);
    }
    template <auto... Js> constexpr auto shuffle(value_seq<Js...>) const noexcept {
      return zs::make_tuple(get<Js>()...);
    }
    template <std::size_t... Js> constexpr auto shuffle(std::index_sequence<Js...>) const noexcept {
      return zs::make_tuple(get<Js>()...);
    }
    /// transform
    template <typename UnaryOp> constexpr auto transform(UnaryOp &&op) const noexcept {
      return zs::make_tuple(op(get<Is>())...);
    }
    ///
    constexpr auto initializer() const noexcept { return std::initializer_list(get<Is>()...); }

    constexpr operator tuple<Ts...>() const noexcept { return *this; }
  };

  template <class... Types> class tuple;

  template <typename... Ts> struct tuple
      : tuple_base<std::index_sequence_for<Ts...>, type_seq<Ts...>> {
    using base_t = tuple_base<std::index_sequence_for<Ts...>, type_seq<Ts...>>;
    using tuple_types = typename base_t::tuple_types;

    constexpr tuple() = default;
    ~tuple() = default;
    template <typename... Vs> constexpr tuple(Vs &&...vs) noexcept : base_t{FWD(vs)...} {}
    constexpr tuple(tuple &&) = default;
    constexpr tuple(const tuple &) = default;
    constexpr tuple &operator=(tuple &&) = default;
    constexpr tuple &operator=(const tuple &) = default;

    // vec
    template <typename VecT>
    constexpr std::enable_if_t<VecT::extent == sizeof...(Ts), tuple &> operator=(
        const VecInterface<VecT> &v) noexcept {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // std::array
    template <typename TT, std::size_t dd> constexpr tuple &operator=(const std::array<TT, dd> &v) {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // c-array
    template <typename Vec>
    constexpr std::enable_if_t<std::is_array_v<Vec>, tuple &> operator=(const Vec &v) {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // std::tuple
    template <typename... Args> constexpr tuple &operator=(const std::tuple<Args...> &tup) {
      assign_impl(FWD(tup),
                  std::make_index_sequence<zs::math::min(sizeof...(Ts), sizeof...(Args))>{});
      return *this;
    }

  private:
    template <typename VecT, std::size_t... Is>
    constexpr void assign_impl(const VecInterface<VecT> &v, index_seq<Is...>) noexcept {
      ((void)(this->template get<Is>() = v.val(Is)), ...);
    }
    template <typename Vec, std::size_t... Is>
    constexpr auto assign_impl(const Vec &v, index_seq<Is...>) noexcept -> decltype(v[0], void()) {
      ((void)(this->template get<Is>() = v[Is]), ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr void assign_impl(const std::tuple<Args...> &tup, index_seq<Is...>) noexcept {
      ((void)(this->template get<Is>() = std::get<Is>(tup)), ...);
    }
  };

  template <typename... Args> tuple(Args...) -> tuple<Args...>;

  template <typename> struct is_std_tuple : std::false_type {};
  template <typename... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};
  template <typename T> static constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

  /** tuple_element */
  template <std::size_t I, typename T, typename = void> struct tuple_element;
  template <std::size_t I, typename... Ts>
  struct tuple_element<I, tuple<Ts...>, std::enable_if_t<(I < sizeof...(Ts))>> {
    using type = select_type<I, typename tuple<Ts...>::tuple_types>;
  };
  template <std::size_t I, typename Tup> using tuple_element_t
      = std::enable_if_t<is_tuple_v<Tup>, std::enable_if_t<(I < (tuple_size_v<Tup>)),
                                                           typename tuple_element<I, Tup>::type>>;

  /** operations */

  /** get */
  template <std::size_t I, typename... Ts>
  constexpr decltype(auto) get(const tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <std::size_t I, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <std::size_t I, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &&t) noexcept {
    return std::move(t).template get<I>();
  }

  template <typename T, typename... Ts>
  constexpr decltype(auto) get(const tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &&t) noexcept {
    return std::move(t).template get<T>();
  }

  /** make_tuple */
  template <typename... Args> constexpr auto make_tuple(Args &&...args) {
    return zs::tuple<special_decay_t<Args>...>{FWD(args)...};
  }
  template <typename T, std::size_t... Is>
  constexpr auto make_uniform_tuple(T &&v, index_seq<Is...>) noexcept {
    return make_tuple((Is ? v : v)...);
  }
  template <std::size_t N, typename T> constexpr auto make_uniform_tuple(T &&v) noexcept {
    return make_uniform_tuple(FWD(v), std::make_index_sequence<N>{});
  }

  /** linear_to_multi */
  template <auto... Ns, std::size_t... Is, enable_if_all<(Ns > 0)...> = 0>
  constexpr auto index_to_coord(std::size_t I, value_seq<Ns...> vs, index_seq<Is...>) {
    constexpr auto N = sizeof...(Ns);
    using Tn = typename value_seq<Ns...>::value_type;
    // using RetT = typename gen_seq<N>::template uniform_types_t<tuple, Tn>;
    constexpr auto exsuf = vs.template scan<2>(multiplies<std::size_t>{});
    Tn bases[N]{exsuf.get_value(wrapv<Is>()).value...};
    Tn cs[N]{};
    for (std::size_t i = 0; i != N; ++i) {
      cs[i] = I / bases[i];
      I -= bases[i] * cs[i];
    }
    return zs::make_tuple(cs[Is]...);
  }
  template <auto... Ns, enable_if_all<(Ns > 0)...> = 0>
  constexpr auto index_to_coord(std::size_t I, value_seq<Ns...> vs) {
    return index_to_coord(I, vs, std::make_index_sequence<sizeof...(Ns)>{});
  }

  /** forward_as_tuple */
  template <typename... Ts> constexpr auto forward_as_tuple(Ts &&...ts) noexcept {
    return zs::tuple<Ts &&...>{FWD(ts)...};
  }

  /** tuple_cat */
  namespace tuple_detail_impl {
    /// concat
    template <typename... Tuples> struct concat {
      static_assert((is_tuple_v<remove_cvref_t<Tuples>> && ...),
                    "concat should only take zs::tuple type template params!");
      using counts = value_seq<remove_cvref_t<Tuples>::tuple_types::count...>;
      static constexpr auto length = counts{}.reduce(plus<std::size_t>{}).value;
      using indices = typename gen_seq<length>::ascend;
      using outer = decltype(
          counts{}.template scan<1, plus<std::size_t>>().map(count_leq{}, wrapv<length>{}));
      using inner = decltype(vseq_t<indices>{}.compwise(
          std::minus<std::size_t>{},
          counts{}.template scan<0, std::plus<std::size_t>>().shuffle(outer{})));
      // using types = decltype(type_seq<typename
      // remove_cvref_t<Tuples>::tuple_types...>{}.shuffle(outer{}).shuffle_join(inner{}));
      template <auto... Os, auto... Is, typename... Tups>
      static constexpr auto get_ret_type(value_seq<Os...>, value_seq<Is...>, Tups &&...tups) {
        auto tup = forward_as_tuple(FWD(tups)...);
        return type_seq<decltype(get<Is>(get<Os>(tup)))...>{};
      }
      // https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat
      using types = decltype(get_ret_type(outer{}, inner{}, std::declval<Tuples>()...));
    };
    template <typename R, auto... Os, auto... Is, typename Tuple>
    constexpr decltype(auto) tuple_cat_impl(value_seq<Os...>, value_seq<Is...>, Tuple &&tup) {
      return R{get<Is>(get<Os>(tup))...};
    }
  }  // namespace tuple_detail_impl

  constexpr auto tuple_cat() noexcept { return tuple<>{}; }

  template <auto... Is, typename... Ts> constexpr auto tuple_cat(value_seq<Is...>, Ts &&...tuples) {
    auto tup = zs::forward_as_tuple(FWD(tuples)...);
    return tuple_cat(get<Is>(tup)...);
  }
  template <typename... Ts> constexpr auto tuple_cat(Ts &&...tuples) {
    if constexpr ((!zs::is_tuple_v<remove_cvref_t<Ts>> || ...)) {
      constexpr auto trans = [](auto &&param) {
        if constexpr (zs::is_tuple_v<RM_CVREF_T(param)>)
          return FWD(param);
        else
          return zs::make_tuple<RM_CVREF_T(param)>(FWD(param));
      };
      return tuple_cat(trans(FWD(tuples))...);
    } else {
      constexpr auto marks = value_seq<(remove_cvref_t<Ts>::tuple_size > 0 ? 1 : 0)...>{};
      if constexpr (marks.reduce(logical_and<bool>{})) {
        // using helper = concat<typename std::remove_reference_t<Ts>::tuple_types...>;
        using helper = tuple_detail_impl::concat<Ts...>;
        return tuple_detail_impl::tuple_cat_impl<assemble_t<tuple, typename helper::types>>(
            typename helper::outer{}, typename helper::inner{},
            zs::forward_as_tuple(FWD(tuples)...));
      } else {
        constexpr auto N = marks.reduce(plus<int>{}).value;
        constexpr auto offsets = marks.scan();  // exclusive scan
        constexpr auto tags = marks.pair(offsets);
        constexpr auto seq
            = tags.filter(typename vseq_t<typename gen_seq<N>::ascend>::template to_iseq<int>{});
        return tuple_cat(seq, FWD(tuples)...);
      }
    }
  }

  template <typename TupA, typename TupB,
            enable_if_t<(is_tuple_v<TupA> || is_tuple_v<remove_cvref_t<TupB>>)> = 0>
  constexpr auto operator+(TupA &&tupa, TupB &&tupb) {
    return tuple_cat(FWD(tupa), FWD(tupb));
  }

  /** apply */
  namespace detail {
    template <class F, class Tuple, std::size_t... Is,
              enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
    constexpr decltype(auto) apply_impl(F &&f, Tuple &&t, index_seq<Is...>) {
      // should use constexpr zs::invoke
      return f(get<Is>(t)...);
    }
  }  // namespace detail
  template <class F, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(F &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              std::make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  template <template <class...> class F, class Tuple,
            enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(assemble_t<F, get_ttal_t<remove_cvref_t<Tuple>>> &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              std::make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  template <std::size_t... Is, typename... Ts>
  constexpr auto shuffle(index_seq<Is...>, const std::tuple<Ts...> &tup) {
    return std::make_tuple(std::get<Is>(tup)...);
  }
  template <std::size_t... Is, typename... Ts>
  constexpr auto shuffle(index_seq<Is...>, const zs::tuple<Ts...> &tup) {
    return zs::make_tuple(zs::get<Is>(tup)...);
  }

  /** tie */
  template <typename... Args> constexpr auto tie(Args &...args) noexcept {
    return zs::tuple<Args &...>{args...};
  }

  /** make_from_tuple */
  namespace tuple_detail_impl {
    template <class T, class Tuple, std::size_t... Is>
    constexpr T make_from_tuple_impl(Tuple &&t, index_seq<Is...>) {
      return T{get<Is>(t)...};
    }
  }  // namespace tuple_detail_impl

  template <class T, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr T make_from_tuple(Tuple &&t) {
    return tuple_detail_impl::make_from_tuple_impl<T>(
        FWD(t), std::make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  // need this because zs::tuple's rvalue deduction not finished
  template <typename T> using capture_t
      = conditional_t<std::is_lvalue_reference<T>{}, std::add_lvalue_reference_t<T>,
                      std::remove_reference_t<T>>;
  template <typename... Ts> constexpr auto fwd_capture(Ts &&...xs) {
    return tuple<capture_t<Ts>...>(FWD(xs)...);
  }
#define FWD_CAPTURE(...) ::zs::fwd_capture(FWD(__VA_ARGS__))

}  // namespace zs

namespace std {
  template <typename... Ts> struct tuple_size<zs::tuple<Ts...>>
      : std::integral_constant<std::size_t, zs::tuple_size_v<zs::tuple<Ts...>>> {};
  template <std::size_t I, typename... Ts> struct tuple_element<I, zs::tuple<Ts...>> {
    using type = zs::tuple_element_t<I, zs::tuple<Ts...>>;
  };
}  // namespace std
