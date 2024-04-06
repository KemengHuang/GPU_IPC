#pragma once
#include <stdint.h>

#include <bitset>
#include <cstring>
#include <limits>

#include "../../TypeAlias.hpp"

namespace zs {

  /// bitwise identical reinterpretation
  [[deprecated("use reinterpret_bits<T>(...) instead.")]] constexpr float int_as_float(
      int i) noexcept {
    static_assert(sizeof(int) == sizeof(float), "int bits != float bits");
    union {
      int i{0};
      float f;
    } u{};
    u.i = i;
    return u.f;
  }
  [[deprecated("use reinterpret_bits<T>(...) instead.")]] constexpr int float_as_int(
      float f) noexcept {
    static_assert(sizeof(int) == sizeof(float), "int bits != float bits");
    union {
      int i{0};
      float f;
    } u{};
    u.f = f;
    return u.i;
  }
  [[deprecated("use reinterpret_bits<T>(...) instead.")]] constexpr double longlong_as_double(
      long long l) noexcept {
    static_assert(sizeof(long long) == sizeof(double), "long long bits != double bits");
    union {
      long long l{0};
      double d;
    } u{};
    u.l = l;
    return u.d;
  }
  [[deprecated("use reinterpret_bits<T>(...) instead.")]] constexpr long long double_as_longlong(
      double d) noexcept {
    static_assert(sizeof(long long) == sizeof(double), "long long bits != double bits");
    union {
      long long l{0};
      double d;
    } u{};
    u.d = d;
    return u.l;
  }
  // ref: https://www.youtube.com/watch?v=_qzMpk-22cc
  // CppCon 2019: Timur Doumler “Type punning in modern C++”

  template <typename DstT, typename SrcT> constexpr auto reinterpret_bits(SrcT &&val) noexcept {
    using Src = std::remove_cv_t<std::remove_reference_t<SrcT>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstT>>;
    static_assert(sizeof(Src) == sizeof(Dst),
                  "Source Type and Destination Type must be of the same size");
    static_assert(std::is_trivially_copyable_v<Src> && std::is_trivially_copyable_v<Dst>,
                  "Both types should be trivially copyable.");
    static_assert(std::alignment_of_v<Src> % std::alignment_of_v<Dst> == 0,
                  "The original type should at least have an alignment as strict.");
    // unsafe due to strict aliasing rule
    // ref: https://en.cppreference.com/w/cpp/language/reinterpret_cast#Type_aliasing
    return reinterpret_cast<Dst const volatile &>(val);
  }
  /// morton code
  constexpr u32 expand_bits_32(u32 v) noexcept {  // expands lower 10-bits to 30 bits
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
  }
  constexpr u64 expand_bits_64(u32 v) noexcept {
    // 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3,
    // 0x1249249249249249
    u64 x = v & 0x1fffff;  // retrieve 21-bits
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
  }
  constexpr u32 compact_bits_32(u32 v) noexcept {
    // 0, 0x000003ff, 0x30000ff, 0x0300f00f, 0x30c30c3, 0x9249249
    v &= 0x9249249;
    v = (v ^ (v >> 2)) & 0x30c30c3;
    v = (v ^ (v >> 4)) & 0x0300f00f;
    v = (v ^ (v >> 8)) & 0x30000ff;
    v = (v ^ (v >> 16)) & 0x000003ff;
    return v;
  }
  constexpr u32 compact_bits_64(u64 v) noexcept {
    // 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3,
    // 0x1249249249249249
    v &= 0x1249249249249249;
    v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3;
    v = (v ^ (v >> 4)) & 0x100f00f00f00f00f;
    v = (v ^ (v >> 8)) & 0x1f0000ff0000ff;
    v = (v ^ (v >> 16)) & 0x1f00000000ffff;
    v = (v ^ (v >> 32)) & 0x1fffff;
    return static_cast<u32>(v);
  }
  constexpr u32 morton_3d_32(float x, float y, float z) noexcept {
    return (expand_bits_32((u32)(x * 1024.f)) << 2) | (expand_bits_32((u32)(y * 1024.f)) << 1)
           | expand_bits_32((u32)(z * 1024.f));
  }
  constexpr u64 morton_3d_64(double x, double y, double z) noexcept {
    return (expand_bits_64((u32)(x * 2097152.)) << 2) | (expand_bits_64((u32)(y * 2097152.)) << 1)
           | expand_bits_64((u32)(z * 2097152.));
  }
  template <typename T> constexpr auto morton_3d(T x, T y, T z) noexcept {
      if constexpr (std::is_same_v<T, float>)
          return morton_3d_32(x, y, z);
      return morton_3d_64(x, y, z);
  }
  template <int dim, typename Vec, std::enable_if_t<(dim == 3), char> = 0>
  constexpr auto morton_code(const Vec &coord) noexcept {
    return morton_3d(coord[0], coord[1], coord[2]);
  }
#if 0
  typename<typename T, int dim, enable_if_t<dim == 2> = 0> constexpr auto morton_code(
      const vec<T, dim> &coord) noexcept {
    return morton_2d<T>(coord[0], coord[1]);
  }
#endif
  template <typename Tn> constexpr Tn round_down(Tn n, Tn alignment) noexcept {
    return n & (~(alignment - (Tn)1));
  }
  template <typename Tn> constexpr Tn round_up(Tn n, Tn alignment) noexcept {
    return (n + alignment - (Tn)1) & (~(alignment - (Tn)1));
  }
  /**
   */
  template <typename Tn = std::size_t> constexpr Tn interleaved_bit_mask(int dim) noexcept {
    constexpr Tn totalBits = sizeof(Tn) << (Tn)3;
    Tn mask = 0;
    for (Tn curBit = 0; curBit < totalBits; curBit += (Tn)dim) mask |= ((Tn)1 << curBit);
    return mask;
  }
  /**
   *	\fn uint32_t bit_length(uint32_t N)
   *	\brief compute the count of significant digits of a number
   *	\param N the number
   */
  template <typename Tn> constexpr Tn bit_length(Tn N) noexcept {
    if (N > (Tn)0)
      return bit_length(N >> (Tn)1) + (Tn)1;
    else
      return (Tn)0;
  }
  /**
   *	\fn uint32_t bit_count(uint32_t N)
   *	\brief compute the count of digits required to express integers in [0, N)
   *  \param N the maximum of the range
   */
  template <typename Tn> constexpr Tn bit_count(Tn N) noexcept {
    if (N > (Tn)0)
      return bit_length(N - (Tn)1);
    else
      return (Tn)0;
  }

  template <typename Tn> constexpr Tn next_2pow(Tn n) noexcept { return (Tn)1 << bit_count(n); }
  /**
   *	\fn uint32_t next_power_of_two(uint32_t i)
   *	\brief compute the next power of two bigger than the number i
   *	\param i the number
   */
  constexpr uint32_t next_power_of_two(uint32_t i) noexcept {
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    return i + 1;
  }

  template <typename Integer>
  constexpr Integer binary_reverse(Integer data, char loc = sizeof(Integer) * 8 - 1) {
    if (data == 0) return 0;
    return ((data & 1) << loc) | binary_reverse(data >> 1, loc - 1);
  }

  template <typename Integer> constexpr unsigned count_leading_zeros(Integer data) {
    unsigned res{0};
    data = binary_reverse(data);
    if (data == 0) return sizeof(Integer) * 8;
    while ((data & 1) == 0) res++, data >>= 1;
    return res;
  }

  constexpr int bit_pack(const uint64_t mask, const uint64_t data) {
    uint64_t slresult = 0;
    uint64_t &ulresult{slresult};
    uint64_t uldata = data;
    int count = 0;
    ulresult = 0;

    uint64_t rmask = binary_reverse(mask);
    unsigned char lz{0};

    while (rmask) {
      lz = static_cast<unsigned char>(count_leading_zeros(rmask));
      uldata >>= lz;
      ulresult <<= 1;
      count++;
      ulresult |= (uldata & 1);
      uldata >>= 1;
      rmask <<= lz + 1;
    }
    ulresult <<= 64 - count;  // 64 means 64 bits ... maybe not use a constant 64 ...
    ulresult = binary_reverse(ulresult);
    return (int)slresult;
  }

  constexpr uint64_t bit_spread(const uint64_t mask, const int data) {
    uint64_t rmask = binary_reverse(mask);
    int dat = data;
    uint64_t result = 0;
    unsigned char lz{0};
    while (rmask) {
      lz = static_cast<unsigned char>(count_leading_zeros(rmask) + 1);
      result = result << lz | (dat & 1);
      dat >>= 1, rmask <<= lz;
    }
    result = binary_reverse(result) >> count_leading_zeros(mask);
    return result;
  }

  ///
  /// borrowed from gvdb_library/kernels/cuda_gvdb.cuh
  ///
  constexpr float float_construct(uint32_t m) noexcept {
    const uint32_t ieeeMantissa = 0x007FFFFFu;  // binary32 mantissa bitmask
    const uint32_t ieeeOne = 0x3F800000u;       // 1.0 in IEEE binary32
    m &= ieeeMantissa;                          // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                               // Add fractional part to 1.0
    float f = reinterpret_bits<float>(m);       // Range [1:2]
    return f - 1.0;                             // Range [0:1]
  }

  constexpr u8 num_bits_on(uint64_t v) noexcept {
    v = v - ((v >> 1) & 0x5555555555555555LLU);
    v = (v & 0x3333333333333333LLU) + ((v >> 2) & (0x3333333333333333LLU));
    return (((v + (v >> 4)) & 0xF0F0F0F0F0F0F0FLLU) * 0x101010101010101LLU) >> 56;
  }

  constexpr bool is_bit_on(uint64_t mask, char b) noexcept {
    return (mask & ((uint64_t)1 << (b & 63))) != 0;
  }

}  // namespace zs
