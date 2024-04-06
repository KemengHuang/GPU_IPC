#pragma once
#include <tuple>
#include <type_traits>
#include <variant>

#include "../meta/Sequence.h"

namespace zs {

  /// https://github.com/SuperV1234/ndctechtown2020/blob/master/7_a_match.pdf
  template <typename... Fs> struct overload_set : Fs... {
    template <typename... Xs> constexpr overload_set(Xs &&...xs) : Fs{std::forward<Xs>(xs)}... {}
    using Fs::operator()...;
  };
  /// class template argument deduction
  template <typename... Xs> overload_set(Xs &&...xs) -> overload_set<remove_cvref_t<Xs>...>;

  template <typename... Fs> constexpr auto make_overload_set(Fs &&...fs) {
    return overload_set<std::decay_t<Fs>...>(std::forward<Fs>(fs)...);
  }

  template <typename... Ts> using variant = std::variant<Ts...>;

  template <typename... Fs> constexpr auto match(Fs &&...fs) {
#if 0
  return [visitor = overload_set{std::forward<Fs>(fs)...}](
             auto &&...vs) -> decltype(auto) {
    return std::visit(visitor, std::forward<decltype(vs)>(vs)...);
  };
#else
    return [visitor = make_overload_set(std::forward<Fs>(fs)...)](auto &&...vs) -> decltype(auto) {
      return std::visit(visitor, std::forward<decltype(vs)>(vs)...);
    };
#endif
  }

  template <typename> struct is_variant : std::false_type {};
  template <typename... Ts> struct is_variant<variant<Ts...>> : std::true_type {};

  template <typename Visitor> struct VariantTaskExecutor {
    Visitor visitor;

    VariantTaskExecutor() = default;
    template <typename F> constexpr VariantTaskExecutor(F &&f) : visitor{FWD(f)} {}

    template <typename Fn, typename... Args> struct CheckCallable {
    private:
      template <typename F, typename... Ts> static constexpr std::false_type test(...) {
        return std::false_type{};
      }
      template <typename F, typename... Ts> static constexpr std::true_type test(
          void_t<decltype(std::declval<Fn>()(std::declval<Args>()...))> *) {
        return std::true_type{};
      }

    public:
      static constexpr bool value = test<Fn, Args...>(nullptr);
    };

    template <std::size_t No, typename Args, std::size_t... Ns, std::size_t i, std::size_t... js,
              std::size_t I, std::size_t... Js>
    constexpr void traverse(bool &tagMatch, Args &args,
                            const std::array<std::size_t, sizeof...(Ns)> &varIndices,
                            index_seq<Ns...> dims, index_seq<i, js...> indices,
                            index_seq<I, Js...>) {
      if constexpr (No == 0) {
        if constexpr (CheckCallable<
                          Visitor,
                          std::variant_alternative_t<I,
                                                     remove_cvref_t<std::tuple_element_t<i, Args>>>,
                          std::variant_alternative_t<
                              Js, remove_cvref_t<std::tuple_element_t<js, Args>>>...>::value) {
          if ((varIndices[i] == I) && ((varIndices[js] == Js) && ...)) {
            tagMatch = true;
            visitor(std::get<I>(std::get<i>(args)), std::get<Js>(std::get<js>(args))...);
            // std::invoke(visitor, std::get<I>(std::get<i>(args)),
            //            std::get<Js>(std::get<js>(args))...);
            return;
          }
        }
      } else {
        traverse<No - 1>(tagMatch, args, varIndices, dims, indices,
                         index_seq<select_indexed_value<No - 1, Ns...>::value - 1, I, Js...>{});
        if (tagMatch) return;
      }
      if constexpr (I > 0) {  // next loop
        traverse<No>(tagMatch, args, varIndices, dims, indices, index_seq<I - 1, Js...>{});
        if (tagMatch) return;
      }
    }

    template <typename... Args> static constexpr bool all_variant() {
      return (is_variant<remove_cvref_t<Args>>::value && ...);
    }

    template <typename... Args>
    constexpr std::enable_if_t<all_variant<Args...>()> operator()(Args &&...args) {
      using variant_sizes = std::index_sequence<std::variant_size_v<remove_cvref_t<Args>>...>;
      constexpr auto narg = sizeof...(Args);
      constexpr auto lastVariantSize
          = std::variant_size_v<select_indexed_type<narg - 1, remove_cvref_t<Args>...>>;
      auto packedArgs = std::forward_as_tuple(FWD(args)...);
      std::array<std::size_t, narg> varIndices{(args.index())...};
      bool tagMatch{false};

      traverse<narg - 1>(tagMatch, packedArgs, varIndices, variant_sizes{},
                         std::index_sequence_for<Args...>{},
                         std::index_sequence<lastVariantSize - 1>{});
    }
  };
  template <typename Visitor> VariantTaskExecutor(Visitor) -> VariantTaskExecutor<Visitor>;

}  // namespace zs
