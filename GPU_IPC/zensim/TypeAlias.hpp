#pragma once
#include <cstdint>
#include <memory>

// #include "Platform.hpp"

namespace zs {

  using uint = unsigned int;
  // signed
  using i8 = std::conditional_t<sizeof(char) == 1, char, int8_t>;
  using i16 = std::conditional_t<sizeof(short) == 2, short, int16_t>;
  using i32 = std::conditional_t<sizeof(int) == 4, int, int32_t>;
  using i64 = std::conditional_t<sizeof(long long int) == 8, long long int,
                                 std::conditional_t<sizeof(long int) == 8, long int, int64_t>>;
  // unsigned
  using u8 = std::conditional_t<sizeof(unsigned char) == 1, unsigned char, uint8_t>;
  using u16 = std::conditional_t<sizeof(unsigned short) == 2, unsigned short, uint16_t>;
  using u32 = std::conditional_t<sizeof(unsigned int) == 4, unsigned int, uint32_t>;
  using u64 = std::conditional_t<
      sizeof(unsigned long long int) == 8, unsigned long long int,
      std::conditional_t<sizeof(unsigned long int) == 8, unsigned long int, uint64_t>>;
  // floating points
  using f32 = float;
  using f64 = double;

  using sint_t = std::make_signed_t<std::size_t>;

  union dat32 {
    f32 f;
    i32 i;
    u32 u;
    template <typename T> constexpr T &cast() noexcept;
    template <typename T> constexpr T cast() const noexcept;
    constexpr f32 &asFloat() noexcept { return f; }
    constexpr i32 &asSignedInteger() noexcept { return i; }
    constexpr u32 &asUnsignedInteger() noexcept { return u; }
    constexpr f32 asFloat() const noexcept { return f; }
    constexpr i32 asSignedInteger() const noexcept { return i; }
    constexpr u32 asUnsignedInteger() const noexcept { return u; }
  };
  template <> constexpr f32 &dat32::cast<f32>() noexcept { return f; }
  template <> constexpr i32 &dat32::cast<i32>() noexcept { return i; }
  template <> constexpr u32 &dat32::cast<u32>() noexcept { return u; }
  template <> constexpr f32 dat32::cast<f32>() const noexcept { return f; }
  template <> constexpr i32 dat32::cast<i32>() const noexcept { return i; }
  template <> constexpr u32 dat32::cast<u32>() const noexcept { return u; }

  union dat64 {
    f64 d;
    i64 l;
    u64 ul;
    template <typename T> constexpr T &cast() noexcept;
    template <typename T> constexpr T cast() const noexcept;
    constexpr f64 &asFloat() noexcept { return d; }
    constexpr i64 &asSignedInteger() noexcept { return l; }
    constexpr u64 &asUnsignedInteger() noexcept { return ul; }
    constexpr f64 asFloat() const noexcept { return d; }
    constexpr i64 asSignedInteger() const noexcept { return l; }
    constexpr u64 asUnsignedInteger() const noexcept { return ul; }
  };
  template <> constexpr f64 &dat64::cast<f64>() noexcept { return d; }
  template <> constexpr i64 &dat64::cast<i64>() noexcept { return l; }
  template <> constexpr u64 &dat64::cast<u64>() noexcept { return ul; }
  template <> constexpr f64 dat64::cast<f64>() const noexcept { return d; }
  template <> constexpr i64 dat64::cast<i64>() const noexcept { return l; }
  template <> constexpr u64 dat64::cast<u64>() const noexcept { return ul; }

  // kokkos::ObservingRawPtr<T>, OptionalRef<T>
  // vsg::ref_ptr<T>
  template <typename T> using RefPtr = ::std::decay_t<T> *;  ///< non-owning reference
  template <typename T> using ConstRefPtr
      = const ::std::decay_t<T> *;  ///< non-owning const reference
  template <typename T> using Holder = ::std::unique_ptr<T>;
  template <typename T> using SharedHolder = ::std::shared_ptr<T>;

  using NodeID = i32;
  using ProcID = char;
  using StreamID = u32;
  using EventID = u32;

  constexpr void do_nothing(...) noexcept {}
  struct do_nothing_op {
    constexpr void operator()(...) noexcept {}
  };

}  // namespace zs

/// lambda capture
/// https://vittorioromeo.info/index/blog/capturing_perfectly_forwarded_objects_in_lambdas.html
#define FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
#define RM_CVREF_T(...) ::std::remove_cv_t<std::remove_reference_t<decltype(__VA_ARGS__)>>

#if ZS_ENABLE_CUDA && defined(__CUDACC__)
#  if defined(ZS_LAMBDA)
#    undef ZS_LAMBDA
#  endif
#  define ZS_LAMBDA __device__
#else
#  if defined(ZS_LAMBDA)
#    undef ZS_LAMBDA
#  endif
#  define ZS_LAMBDA
#endif

#if ZS_ENABLE_CUDA && defined(__CUDACC__)
#  if defined(ZS_FUNCTION)
#    undef ZS_FUNCTION
#  endif
#  define ZS_FUNCTION __forceinline__ __device__
#else
#  if defined(ZS_FUNCTION)
#    undef ZS_FUNCTION
#  endif
#  define ZS_FUNCTION inline
#endif