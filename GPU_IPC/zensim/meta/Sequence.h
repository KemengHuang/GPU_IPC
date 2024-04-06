#pragma once

#include <type_traits>
#include <utility>

#include "../TypeAlias.hpp"
#include "ControlFlow.h"
#include "Functional.h"
#include "Meta.h"

namespace zs {

  template <typename Tn, Tn... Ns> using integer_seq = std::integer_sequence<Tn, Ns...>;
  template <auto... Ns> using index_seq = std::index_sequence<Ns...>;
  template <auto... Ns> using sindex_seq = std::integer_sequence<sint_t, Ns...>;

  /// indexable type list to avoid recursion
  namespace type_impl {
    template <std::size_t I, typename T> struct indexed_type {
      using type = T;
      static constexpr std::size_t value = I;
    };
    template <typename, typename... Ts> struct indexed_types;

    template <std::size_t... Is, typename... Ts> struct indexed_types<index_seq<Is...>, Ts...>
        : indexed_type<Is, Ts>... {};

    // use pointer rather than reference as in taocpp! [incomplete type error]
    template <std::size_t I, typename T> indexed_type<I, T> extract_type(indexed_type<I, T> *);
    template <typename T, std::size_t I> indexed_type<I, T> extract_index(indexed_type<I, T> *);
  }  // namespace type_impl

  /******************************************************************/
  /** declaration: monoid_op, gen_seq, gather */
  /******************************************************************/

  /// generate index sequence declaration
  template <typename> struct gen_seq_impl;
  template <std::size_t N> using gen_seq = gen_seq_impl<std::make_index_sequence<N>>;

  /// seq + op
  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;

  template <typename... Seqs> struct concat;
  template <typename... Seqs> using concat_t = typename concat<Seqs...>::type;

  template <typename> struct is_type_seq : std::false_type {};
  template <typename... Ts> struct is_type_seq<type_seq<Ts...>> : std::true_type {};
  template <typename SeqT> static constexpr bool is_type_seq_v = is_type_seq<SeqT>::value;

  /******************************************************************/
  /** definition: monoid_op, type_seq, value_seq, gen_seq, gather */
  /******************************************************************/
  /// static uniform non-types
  template <std::size_t... Is> struct gen_seq_impl<index_seq<Is...>> {
    /// arithmetic sequences
    template <auto N0 = 0, auto Step = 1> using arithmetic = index_seq<static_cast<std::size_t>(
        static_cast<sint_t>(N0) + static_cast<sint_t>(Is) * static_cast<sint_t>(Step))...>;
    using ascend = arithmetic<0, 1>;
    using descend = arithmetic<sizeof...(Is) - 1, -1>;
    template <auto J> using uniform = integer_seq<decltype(J), (Is == Is ? J : J)...>;
    template <auto J> using uniform_vseq = value_seq<(Is == Is ? J : J)...>;
    /// types with uniform type/value params
    template <template <typename...> typename T, typename Arg> using uniform_types_t
        = T<std::enable_if_t<Is >= 0, Arg>...>;
    template <template <typename...> typename T, typename Arg>
    static constexpr auto uniform_values(const Arg &arg) {
      return uniform_types_t<T, Arg>{((void)Is, arg)...};
    }
    template <template <auto...> typename T, auto Arg> using uniform_values_t
        = T<(Is >= 0 ? Arg : 0)...>;
  };

  /// type_seq
  template <typename... Ts> struct type_seq {
    static constexpr bool is_value_sequence() noexcept {
      if constexpr (sizeof...(Ts) == 0)
        return true;
      else
        return (is_value_wrapper_v<Ts> && ...);
    }
    static constexpr bool all_values = is_value_sequence();

    using indices = std::index_sequence_for<Ts...>;

    static constexpr auto count = sizeof...(Ts);

    // type
    template <std::size_t I> using type = typename decltype(type_impl::extract_type<I>(
        std::add_pointer_t<type_impl::indexed_types<indices, Ts...>>{}))::type;

    // index
    template <typename, typename = void> struct locator {
      using index = integral_t<std::size_t, limits<std::size_t>::max()>;
    };
    template <typename T> static constexpr std::size_t count_occurencies() noexcept {
      if constexpr (sizeof...(Ts) == 0)
        return 0;
      else
        return (static_cast<std::size_t>(is_same_v<T, Ts>) + ...);
    }
    template <typename T> using occurencies_t = wrapv<count_occurencies<T>()>;
    template <typename T> struct locator<T, std::enable_if_t<count_occurencies<T>() == 1>> {
      using index
          = integral_t<std::size_t,
                       decltype(type_impl::extract_index<T>(
                           std::add_pointer_t<type_impl::indexed_types<indices, Ts...>>{}))::value>;
    };
    template <typename T> using index = typename locator<T>::index;
    ///
    /// operations
    ///
    template <auto I = 0> constexpr auto get_type(wrapv<I> = {}) const noexcept {
      return wrapt<type<I>>{};
    }
    template <typename T = void> constexpr auto get_index(wrapt<T> = {}) const noexcept {
      return index<T>{};
    }
    template <typename Ti, Ti... Is>
    constexpr auto filter(integer_seq<Ti, Is...>) const noexcept {  // for tuple_cat
      return value_seq<index<type_seq<integral_t<Ti, 1>, integral_t<Ti, Is>>>::value...>{};
    }
    template <typename... Args> constexpr auto pair(type_seq<Args...>) const noexcept;
    template <auto... Is> constexpr auto shuffle(value_seq<Is...>) const noexcept {
      return type_seq<typename type_seq<Ts...>::template type<Is>...>{};
    }
    template <typename Ti, Ti... Is> constexpr auto shuffle(integer_seq<Ti, Is...>) const noexcept {
      return shuffle(value_seq<Is...>{});
    }
    template <auto... Is> constexpr auto shuffle_join(value_seq<Is...>) const noexcept {
      return type_seq<typename Ts::template type<Is>...>{};
    }
    template <typename Ti, Ti... Is>
    constexpr auto shuffle_join(integer_seq<Ti, Is...>) const noexcept {
      return shuffle_join(value_seq<Is...>{});
    }
  };
  // template <typename... Ts> struct type_seq<type_seq<Ts...>> : type_seq<Ts...> {};
  template <typename T> struct is_tseq : std::false_type {};
  template <typename... Ts> struct is_tseq<type_seq<Ts...>> : std::true_type {};

  /// select type by index
  template <std::size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <std::size_t I, typename... Ts> using select_indexed_type
      = select_type<I, type_seq<Ts...>>;

  template <typename TypeSeq, typename Indices> using shuffle_t
      = decltype(INST_(TypeSeq).shuffle(INST_(Indices)));
  template <typename TypeSeq, typename Indices> using shuffle_join_t
      = decltype(INST_(TypeSeq).shuffle_join(INST_(Indices)));

  /// value_seq
  template <auto... Ns> struct value_seq : type_seq<integral_t<decltype(Ns), Ns>...> {
    using base_t = type_seq<integral_t<decltype(Ns), Ns>...>;
    using indices = typename base_t::indices;
    static constexpr auto count = base_t::count;
    static constexpr auto get_common_type() noexcept {
      if constexpr (base_t::count == 0)
        return wrapt<std::size_t>{};
      else
        return wrapt<std::common_type_t<decltype(Ns)...>>{};
    }
    using value_type = typename decltype(get_common_type())::type;
    using iseq = integer_seq<value_type, (value_type)Ns...>;
    template <typename T> using to_iseq = integer_seq<T, (T)Ns...>;

    template <std::size_t I> static constexpr auto value = base_t::template type<I>::value;

    value_seq() noexcept = default;
    template <typename Ti, auto cnt = count, enable_if_t<(cnt > 0)> = 0>
    constexpr value_seq(integer_seq<Ti, Ns...>) noexcept {}
    template <auto cnt = count, enable_if_t<(cnt > 0)> = 0>
    constexpr value_seq(wrapv<Ns>...) noexcept {}
    ///
    /// operations
    ///
    template <auto I = 0> constexpr auto get_value(wrapv<I> = {}) const noexcept {
      return typename base_t::template type<I>{};
    }
    template <typename Ti = value_type> constexpr auto get_iseq(wrapt<Ti> = {}) const noexcept {
      return integer_seq<Ti, (Ti)Ns...>{};
    }
    template <typename BinaryOp> constexpr auto reduce(BinaryOp) const noexcept {
      return wrapv<monoid_op<BinaryOp>{}(Ns...)>{};
    }
    template <typename UnaryOp, typename BinaryOp>
    constexpr auto reduce(UnaryOp, BinaryOp) const noexcept {
      return wrapv<monoid_op<BinaryOp>{}(UnaryOp{}(Ns)...)>{};
    }
    template <typename UnaryOp, typename BinaryOp, std::size_t... Is>
    constexpr auto map_reduce(UnaryOp, BinaryOp, index_seq<Is...> = indices{}) noexcept {
      return wrapv<monoid_op<BinaryOp>{}(UnaryOp{}(Is, Ns)...)>{};
    }
    template <typename BinaryOp, auto... Ms>
    constexpr auto compwise(BinaryOp, value_seq<Ms...>) const noexcept {
      return value_seq<BinaryOp{}(Ns, Ms)...>{};
    }
    /// map (Op(i), index_sequence)
    template <typename MapOp, auto... Js>
    constexpr auto map(MapOp, value_seq<Js...>) const noexcept {
      return value_seq<MapOp{}(Js, Ns...)...>{};
    }
    template <typename MapOp, typename Ti, Ti... Js>
    constexpr auto map(MapOp &&op, integer_seq<Ti, Js...>) const noexcept {
      return map(FWD(op), value_seq<Js...>{});
    }
    template <typename MapOp, auto N> constexpr auto map(MapOp &&op, wrapv<N> = {}) const noexcept {
      return map(FWD(op), std::make_index_sequence<N>{});
    }
    /// cat
    template <auto... Is> constexpr auto concat(value_seq<Is...>) const noexcept {
      return value_seq<Ns..., Is...>{};
    }
    template <typename Ti, Ti... Is> constexpr auto concat(integer_seq<Ti, Is...>) const noexcept {
      return value_seq<Ns..., Is...>{};
    }
    /// shuffle
    constexpr auto shuffle(value_seq<>) const noexcept { return value_seq<>{}; }
    template <auto... Is> constexpr auto shuffle(value_seq<Is...>) const noexcept {
      return value_seq<base_t::template type<Is>::type::value...>{};
    }
    template <typename Ti, Ti... Is> constexpr auto shuffle(integer_seq<Ti, Is...>) const noexcept {
      return shuffle(value_seq<Is...>{});
    }
    /// transform
    template <typename UnaryOp> constexpr auto transform(UnaryOp) const noexcept {
      return value_seq<UnaryOp{}(Ns)...>{};
    }
    /// for_each
    template <typename F> constexpr void for_each(F &&f) const noexcept { (f(Ns), ...); }
    /// scan
    template <auto Cate, typename BinaryOp, std::size_t... Is>
    constexpr auto scan_impl(BinaryOp, index_seq<Is...>) const noexcept {
      constexpr auto get_sum = [](auto I_) noexcept {
        constexpr auto I = decltype(I_)::value;
        if constexpr (Cate == 0)
          return wrapv<monoid_op<BinaryOp>{}((Is < I ? Ns : monoid_op<BinaryOp>::e)...)>{};
        else if constexpr (Cate == 1)
          return wrapv<monoid_op<BinaryOp>{}((Is <= I ? Ns : monoid_op<BinaryOp>::e)...)>{};
        else if constexpr (Cate == 2)
          return wrapv<monoid_op<BinaryOp>{}((Is > I ? Ns : monoid_op<BinaryOp>::e)...)>{};
        else
          return wrapv<monoid_op<BinaryOp>{}((Is >= I ? Ns : monoid_op<BinaryOp>::e)...)>{};
      };
      return value_seq<RM_CVREF_T(get_sum(wrapv<Is>{}))::value...>{};
    }
    template <auto Cate = 0, typename BinaryOp = plus<value_type>>
    constexpr auto scan(BinaryOp bop = {}) const noexcept {
      return scan_impl<Cate>(bop, indices{});
    }
  };
  template <typename Ti, Ti... Ns> value_seq(integer_seq<Ti, Ns...>) -> value_seq<Ns...>;
  template <auto... Ns> value_seq(wrapv<Ns>...) -> value_seq<Ns...>;

  template <typename T> struct is_vseq : std::false_type {};
  template <auto... Ns> struct is_vseq<value_seq<Ns...>> : std::true_type {};

  template <typename> struct vseq;
  template <auto... Ns> struct vseq<value_seq<Ns...>> { using type = value_seq<Ns...>; };
  template <typename Ti, Ti... Ns> struct vseq<integer_seq<Ti, Ns...>> {
    using type = value_seq<Ns...>;
  };
  template <typename Ti, Ti N> struct vseq<integral_t<Ti, N>> { using type = value_seq<N>; };
  template <typename Seq> using vseq_t = typename vseq<Seq>::type;

  /// select (constant integral) value (integral_constant<T, N>) by index
  template <std::size_t I, typename ValueSeq> using select_value =
      typename ValueSeq::template type<I>;
  template <std::size_t I, auto... Ns> using select_indexed_value
      = select_value<I, value_seq<Ns...>>;

  /** utilities */
  // extract (type / non-type) template argument list
  template <typename T> struct get_ttal;  // extract type template parameter list
  template <template <class...> class T, typename... Args> struct get_ttal<T<Args...>> {
    using type = type_seq<Args...>;
  };
  template <typename T> using get_ttal_t = typename get_ttal<T>::type;

  template <typename T> struct get_nttal;  // extract type template parameter list
  template <template <auto...> class T, auto... Args> struct get_nttal<T<Args...>> {
    using type = value_seq<Args...>;
  };
  template <typename T> using get_nttal_t = typename get_nttal<T>::type;

  /// assemble functor given template argument list
  /// note: recursively unwrap type_seq iff typelist size is one
  template <template <class...> class T, typename... Args> struct assemble {
    using type = T<Args...>;
  };
  template <template <class...> class T, typename... Args> struct assemble<T, type_seq<Args...>>
      : assemble<T, Args...> {};
  template <template <class...> class T, typename... Args> using assemble_t =
      typename assemble<T, Args...>::type;

  /// same functor with a different template argument list
  template <typename...> struct alternative;
  template <template <class...> class T, typename... Ts, typename... Args>
  struct alternative<T<Ts...>, type_seq<Args...>> {
    using type = T<Args...>;
  };
  template <template <class...> class T, typename... Ts, typename... Args>
  struct alternative<T<Ts...>, Args...> {
    using type = T<Args...>;
  };
  template <typename TTAT, typename... Args> using alternative_t =
      typename alternative<TTAT, Args...>::type;

  /// concatenation
  namespace detail {
    struct concatenation_op {
      template <typename... As, typename... Bs>
      constexpr auto operator()(type_seq<As...>, type_seq<Bs...>) const noexcept {
        return type_seq<As..., Bs...>{};
      }
      template <typename... SeqT> constexpr auto operator()(type_seq<SeqT...>) const noexcept {
        constexpr auto seq_lambda = [](auto I_) noexcept {
          using T = select_indexed_type<decltype(I_)::value, SeqT...>;
          return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
        };
        constexpr std::size_t N = sizeof...(SeqT);

        if constexpr (N == 0)
          return type_seq<>{};
        else if constexpr (N == 1)
          return seq_lambda(index_c<0>);
        else if constexpr (N == 2)
          return (*this)(seq_lambda(index_c<0>), seq_lambda(index_c<1>));
        else {
          constexpr std::size_t halfN = N / 2;
          return (*this)((*this)(type_seq<SeqT...>{}.shuffle(typename gen_seq<halfN>::ascend{})),
                         (*this)(type_seq<SeqT...>{}.shuffle(
                             typename gen_seq<N - halfN>::template arithmetic<halfN>{})));
        }
      }
    };
  }  // namespace detail
  template <typename... TSeqs> using concatenation_t
      = decltype(std::declval<detail::concatenation_op>()(type_seq<TSeqs...>{}));

  template <typename... Ts> template <typename... Args>
  constexpr auto type_seq<Ts...>::pair(type_seq<Args...>) const noexcept {
    return type_seq<concatenation_t<Ts, Args>...>{};
  }

  /// permutation / combination
  namespace detail {
    struct compose_op {
      // merger
      template <typename... As, typename... Bs, std::size_t... Is>
      constexpr auto get_seq(type_seq<As...>, type_seq<Bs...>, index_seq<Is...>) const noexcept {
        constexpr auto Nb = sizeof...(Bs);
        return type_seq<concatenation_t<select_indexed_type<Is / Nb, As...>,
                                        select_indexed_type<Is % Nb, Bs...>>...>{};
      }
      template <typename... As, typename... Bs>
      constexpr auto operator()(type_seq<As...>, type_seq<Bs...>) const noexcept {
        constexpr auto N = (sizeof...(As)) * (sizeof...(Bs));
        return conditional_t<N == 0, type_seq<>,
                             decltype(get_seq(type_seq<As...>{}, type_seq<Bs...>{},
                                              std::make_index_sequence<N>{}))>{};
      }

      /// more general case
      template <typename... SeqT> constexpr auto operator()(type_seq<SeqT...>) const noexcept {
        constexpr auto seq_lambda = [](auto I_) noexcept {
          using T = select_indexed_type<decltype(I_)::value, SeqT...>;
          return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
        };
        constexpr std::size_t N = sizeof...(SeqT);
        if constexpr (N == 0)
          return type_seq<>{};
        else if constexpr (N == 1)
          return map_t<type_seq, decltype(seq_lambda(index_c<0>))>{};
        else if constexpr (N == 2)
          return (*this)(seq_lambda(index_c<0>), seq_lambda(index_c<1>));
        else if constexpr (N > 2) {
          constexpr std::size_t halfN = N / 2;
          return (*this)((*this)(type_seq<SeqT...>{}.shuffle(typename gen_seq<halfN>::ascend{})),
                         (*this)(type_seq<SeqT...>{}.shuffle(
                             typename gen_seq<N - halfN>::template arithmetic<halfN>{})));
        }
      }
    };
  }  // namespace detail
  template <typename... TSeqs> using compose_t
      = decltype(detail::compose_op{}(type_seq<TSeqs...>{}));

  /// join

  /// sequence manipulation declaration
  template <typename, typename> struct gather;
  /// uniform value sequence
  template <std::size_t... Is, typename T, T... Ns>
  struct gather<index_seq<Is...>, integer_seq<T, Ns...>> {
    using type = integer_seq<T, select_indexed_value<Is, Ns...>{}...>;
  };
  template <typename Indices, typename ValueSeq> using gather_t =
      typename gather<Indices, ValueSeq>::type;

  /// variadic type template parameters
  template <typename T, template <typename...> class Ref> struct is_type_specialized
      : std::false_type {};
  template <template <typename...> class Ref, typename... Ts>
  struct is_type_specialized<Ref<Ts...>, Ref> : std::true_type {};

  /// variadic non-type template parameters
  template <typename T, template <auto...> class Ref> struct is_value_specialized
      : std::false_type {};
  template <template <auto...> class Ref, auto... Args>
  struct is_value_specialized<Ref<Args...>, Ref> : std::true_type {};

  /** direct operations on sequences */
  template <typename> struct seq_tail { using type = index_seq<>; };
  template <std::size_t I, std::size_t... Is> struct seq_tail<index_seq<I, Is...>> {
    using type = index_seq<Is...>;
  };
  template <typename Seq> using seq_tail_t = typename seq_tail<Seq>::type;

  /** placeholder */
  namespace index_literals {
    // ref: numeric UDL
    // Embracing User Defined Literals Safely for Types that Behave as though Built-in
    // Pablo Halpern
    template <auto partial> constexpr auto index_impl() noexcept { return partial; }
    template <auto partial, char c0, char... c> constexpr auto index_impl() noexcept {
      if constexpr (c0 == '\'')
        return index_impl<partial, c...>();
      else {
        using Tn = decltype(partial);
        static_assert(c0 >= '0' && c0 <= '9', "Invalid non-numeric character");
        static_assert(partial <= (limits<Tn>::max() - (c0 - '0')) / 10, "numeric literal overflow");
        return index_impl<partial *(Tn)10 + (Tn)(c0 - '0'), c...>();
      }
    }

    template <char... c> constexpr auto operator""_th() noexcept {
      constexpr auto id = index_impl<(std::size_t)0, c...>();
      return index_c<id>;
    }
  }  // namespace index_literals

  namespace placeholders {
    using placeholder_type = std::size_t;
    constexpr auto _0 = integral_t<placeholder_type, 0>{};
    constexpr auto _1 = integral_t<placeholder_type, 1>{};
    constexpr auto _2 = integral_t<placeholder_type, 2>{};
    constexpr auto _3 = integral_t<placeholder_type, 3>{};
    constexpr auto _4 = integral_t<placeholder_type, 4>{};
    constexpr auto _5 = integral_t<placeholder_type, 5>{};
    constexpr auto _6 = integral_t<placeholder_type, 6>{};
    constexpr auto _7 = integral_t<placeholder_type, 7>{};
    constexpr auto _8 = integral_t<placeholder_type, 8>{};
    constexpr auto _9 = integral_t<placeholder_type, 9>{};
    constexpr auto _10 = integral_t<placeholder_type, 10>{};
    constexpr auto _11 = integral_t<placeholder_type, 11>{};
    constexpr auto _12 = integral_t<placeholder_type, 12>{};
    constexpr auto _13 = integral_t<placeholder_type, 13>{};
    constexpr auto _14 = integral_t<placeholder_type, 14>{};
    constexpr auto _15 = integral_t<placeholder_type, 15>{};
    constexpr auto _16 = integral_t<placeholder_type, 16>{};
    constexpr auto _17 = integral_t<placeholder_type, 17>{};
    constexpr auto _18 = integral_t<placeholder_type, 18>{};
    constexpr auto _19 = integral_t<placeholder_type, 19>{};
    constexpr auto _20 = integral_t<placeholder_type, 20>{};
    constexpr auto _21 = integral_t<placeholder_type, 21>{};
    constexpr auto _22 = integral_t<placeholder_type, 22>{};
    constexpr auto _23 = integral_t<placeholder_type, 23>{};
    constexpr auto _24 = integral_t<placeholder_type, 24>{};
    constexpr auto _25 = integral_t<placeholder_type, 25>{};
    constexpr auto _26 = integral_t<placeholder_type, 26>{};
    constexpr auto _27 = integral_t<placeholder_type, 27>{};
    constexpr auto _28 = integral_t<placeholder_type, 28>{};
    constexpr auto _29 = integral_t<placeholder_type, 29>{};
    constexpr auto _30 = integral_t<placeholder_type, 30>{};
    constexpr auto _31 = integral_t<placeholder_type, 31>{};
  }  // namespace placeholders
  using place_id = placeholders::placeholder_type;

}  // namespace zs
