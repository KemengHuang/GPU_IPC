#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "VecInterface.hpp"
#include "../math/MathUtils.h"
#include "../meta/Meta.h"
#include "../meta/Relationship.h"
#include "../meta/Sequence.h"
#include "../types/Tuple.h"

namespace zs {

  /// declarations
  template <typename T, typename Extents> struct vec_view;
  template <typename T, typename Extents> struct vec_impl;

  template <typename T, auto... Ns> using vec = vec_impl<T, integer_seq<int, Ns...>>;
  template <typename T, typename Tn, Tn... Ns> using vec_t = vec_impl<T, integer_seq<Tn, Ns...>>;

  /// vec without lifetime managing
  template <typename T, typename Tn, Tn... Ns> struct vec_view<T, integer_seq<Tn, Ns...>>
      : VecInterface<vec_view<T, integer_seq<Tn, Ns...>>> {
    using base_t = VecInterface<vec_view<T, integer_seq<Tn, Ns...>>>;
    // essential defs for any VecInterface
    using value_type = std::remove_const_t<T>;
    using index_type = Tn;
    using indexer_type = indexer<index_type, Ns...>;
    using extents = integer_seq<index_type, Ns...>;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename OtherT, typename IndicesT> using variant_vec = vec_impl<OtherT, IndicesT>;

    constexpr vec_view() = delete;
    constexpr vec_view(const vec_view &) = delete;             // prevents accidental copy of view
    constexpr vec_view &operator=(const vec_view &) = delete;  // prevents accidental copy of view
    constexpr explicit vec_view(T *ptr) : _data{ptr} {}

    constexpr explicit operator variant_vec<value_type, extents>() const noexcept {
      variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = this->val(i);
      return r;
    }

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1) {
        return _data[index];
      } else {
        using R = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>;
        return R{_data + indexer_type::offset(index)};
      }
    }
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1) {
        return _data[index];
      } else {
        using R = vec_view<std::add_const_t<T>,
                           gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>;
        return R{_data + indexer_type::offset(index)};
      }
    }
    template <typename Index> constexpr T &do_val(Index index) noexcept { return _data[index]; }
    template <typename Index> constexpr const T &do_val(Index index) const noexcept {
      return _data[index];
    }

    T *_data;
  };

  /// vec
  template <typename T, typename Tn, Tn... Ns> struct vec_impl<T, std::integer_sequence<Tn, Ns...>>
      : VecInterface<vec_impl<T, std::integer_sequence<Tn, Ns...>>> {
    // static_assert(std::is_trivial<T>::value,
    //              "Vec element type is not trivial!\n");
    using base_t = VecInterface<vec_impl<T, std::integer_sequence<Tn, Ns...>>>;
    using value_type = T;
    using index_type = Tn;
    using indexer_type = indexer<index_type, Ns...>;
    using extents = integer_seq<index_type, Ns...>;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename OtherT, typename IndicesT> using variant_vec = vec_impl<OtherT, IndicesT>;

    T _data[extent];

  public:
    /// expose internal
    constexpr auto data() noexcept -> T * { return _data; }
    constexpr auto data() volatile noexcept -> volatile T * { return (volatile T *)_data; }
    constexpr auto data() const noexcept -> const T * { return _data; }

    /// think this does not break rule of five
    constexpr vec_impl() noexcept = default;
    template <typename... Ts,
              enable_if_all<(sizeof...(Ts) <= extent),
                            (std::is_convertible_v<remove_cvref_t<Ts>, value_type> && ...)> = 0>
    constexpr vec_impl(Ts &&...ts) noexcept : _data{(value_type)ts...} {}
    /// https://github.com/kokkos/kokkos/issues/177
#if 0
    constexpr volatile vec_impl &operator=(const vec_impl &o) volatile {
      for (Tn i = 0; i != extent; ++i) data()[i] = o.data()[i];
      return *this;
    }
#endif
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = std::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    constexpr vec_impl &operator=(const std::tuple<Args...> &tup) {
      *this = from_tuple(tup);
      return *this;
    }
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = zs::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }

    static constexpr vec_impl from_array(const std::array<T, extent> &arr) noexcept {
      vec_impl r{};
      for (Tn i = 0; i != extent; ++i) r.val(i) = arr[i];
      return r;
    }
    constexpr std::array<T, extent> to_array() const noexcept {
      std::array<T, extent> r{};
      for (Tn i = 0; i != extent; ++i) r[i] = _data[i];
      return r;
    }
    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index,
              typename R
              = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) noexcept {
      return R{_data + indexer_type::offset(index)};
    }
    template <typename Index,
              typename R
              = vec_view<std::add_const_t<T>,
                         gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) const noexcept {
      return R{_data + indexer_type::offset(index)};
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr T &operator[](Index index) noexcept {
      return _data[index];
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr const T &operator[](Index index) const noexcept {
      return _data[index];
    }
    template <typename Index> constexpr T &do_val(Index index) noexcept { return _data[index]; }
    template <typename Index> constexpr const T &do_val(Index index) const noexcept {
      return _data[index];
    }
    ///
    template <typename TT> constexpr explicit operator vec_impl<TT, extents>() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx != extent; ++idx) r.val(idx) = _data[idx];
      return r;
    }
  };

  /// make vec
  template <typename... Args, enable_if_all<(!is_std_tuple<remove_cvref_t<Args>>::value, ...)> = 0>
  constexpr auto make_vec(Args &&...args) noexcept {
    using Tn = math::op_result_t<remove_cvref_t<Args>...>;
    return vec<Tn, sizeof...(Args)>{FWD(args)...};
  }
  /// make vec from std tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const std::tuple<Ts...> &tup,
                                                  index_seq<Is...>) noexcept {
    return vec<T, (sizeof...(Ts))>{std::get<Is>(tup)...};
  }
  template <typename T, typename... Ts>
  constexpr auto make_vec(const std::tuple<Ts...> &tup) noexcept {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }
  /// make vec from zs tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const tuple<Ts...> &tup,
                                                  index_seq<Is...>) noexcept {
    return vec<T, (sizeof...(Ts))>{get<Is>(tup)...};
  }
  template <typename T, typename... Ts> constexpr auto make_vec(const tuple<Ts...> &tup) noexcept {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }

  /// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
  /// vector(vec+{0}) homogeneous coordinates

  template <typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<math::op_result_t<remove_cvref_t<Args>...>, sizeof...(Args)>{FWD(args)...};
  }
  template <typename RetT, typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<RetT, sizeof...(Args)>{FWD(args)...};
  }

  template <typename Tn, int dim>
  constexpr auto unpack_coord(const vec<Tn, dim> &id, Tn side_length) {
    using T = std::make_signed_t<Tn>;
    auto bid = id;
    for (int d = 0; d != dim; ++d) bid[d] += (id[d] < 0 ? ((T)1 - (T)side_length) : 0);
    bid = bid / side_length;
    return std::make_tuple(bid, id - bid * side_length);
  }
  template <int dim, typename Tn, typename Ti>
  constexpr auto linear_to_coord(Tn offset, Ti sideLength) {
    using T = math::op_result_t<Tn, Ti>;
    vec<Tn, dim> ret{};
    for (int d = dim - 1; d != -1 && offset; --d, offset = (T)offset / (T)sideLength)
      ret[d] = (T)offset % (T)sideLength;
    return ret;
  }

}  // namespace zs

#include "MatrixUtils.inl"