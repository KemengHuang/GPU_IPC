#pragma once

/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <string>
#include <typeindex>
#include <typeinfo>
#include <array>

namespace zs {

  template <typename T> constexpr const char *type_name() noexcept { return typeid(T).name(); }

  template <typename T> constexpr const char *type_name(const T &) noexcept {
    return type_name<T>();
  }

  template <> constexpr const char *type_name<std::string>() noexcept { return "string"; }
  template <> constexpr const char *type_name<bool>() noexcept { return "char"; }
  template <> constexpr const char *type_name<char>() noexcept { return "char"; }
  template <> constexpr const char *type_name<unsigned char>() noexcept { return "uchar"; }
  template <> constexpr const char *type_name<short>() noexcept { return "short"; }
  template <> constexpr const char *type_name<unsigned short>() noexcept { return "ushort"; }
  template <> constexpr const char *type_name<int>() noexcept { return "int"; }
  template <> constexpr const char *type_name<unsigned int>() noexcept { return "uint"; }
  template <> constexpr const char *type_name<float>() noexcept { return "float"; }
  template <> constexpr const char *type_name<double>() noexcept { return "double"; }

  /// compile-time type inspection
  template <class T> class that_type;
  template <class T> void name_that_type(T &param) {
    // forgot where I picked up this trick...
    that_type<T> tType;
    that_type<decltype(param)> paramType;
  }

  namespace detail {
    template <typename T> constexpr auto get_var_type_str_helper(T &&) noexcept {
#if defined(_MSC_VER)
      return __FUNCSIG__;
#else
      return __PRETTY_FUNCTION__;
#endif
    }

    template <typename T> constexpr auto get_type_str_helper() noexcept {
#if defined(_MSC_VER)
      return __FUNCSIG__;
#else
      return __PRETTY_FUNCTION__;
#endif
    }

    constexpr std::size_t get_type_len_helper(const char *p = nullptr) noexcept {
      if (p == nullptr) return (std::size_t)0;
      std::size_t i = 0;
      for (; p[i]; ++i)
        ;
      return i;
    }

    struct range_pair {
      std::size_t l{}, r{};
    };
    constexpr range_pair locate_char_in_str_helper(const char *p, const char lc,
                                                   const char rc) noexcept {
      if (p == nullptr) return range_pair{0, 0};
      std::size_t l{0};
      for (; *p; ++p, ++l)
        if (*p == lc) break;
      std::size_t r{l + 1}, cnt{1};
      for (++p; *p; ++p, ++r) {
        if (*p == lc)
          cnt++;
        else if (*p == rc)
          cnt--;
        if (cnt == 0) break;
      }
      /// [l, r]
      return range_pair{l, r};
    }

    template <std::size_t head = 0, std::size_t length = 0, typename T>
    constexpr auto get_var_type_substr(T &&v) noexcept {
      constexpr auto typestr = get_type_str_helper<T>();
      using CharT = std::remove_const_t<std::remove_pointer_t<decltype(typestr)>>;
      constexpr auto typelength = get_type_len_helper(typestr);
      static_assert(typelength > head, "sub-string should not exceed the whole string!");
      constexpr auto substrLength
          = (length == 0 ? typelength - head
                         : (length < (typelength - head) ? length : (typelength - head)));
      std::array<CharT, substrLength> ret{};
      for (std::size_t i = 0; i != substrLength; ++i) ret[i] = typestr[i + head];
      return ret;
    }
  }  // namespace detail

  template <typename T> constexpr auto get_type() noexcept {
    constexpr auto typestr = detail::get_type_str_helper<T>();
    using CharT = std::remove_const_t<std::remove_pointer_t<decltype(typestr)>>;
    // constexpr auto typelength = detail::get_type_len_helper(typestr);

#if defined(_MSC_VER)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '<', '>');
    constexpr std::size_t head{pair.l + 1};
    constexpr std::size_t length{pair.r - head};
#elif defined(__clang__)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '[', ']');
    constexpr std::size_t head{pair.l + 5};
    constexpr std::size_t length{pair.r - head};
#elif defined(__GNUC__)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '[', ']');
    constexpr std::size_t head{pair.l + 10};
    constexpr std::size_t length{pair.r - head};
#endif

    std::array<CharT, length> ret{};
    for (std::size_t i = 0; i != length; ++i) ret[i] = typestr[i + head];
    return ret;
  }
  template <typename T> constexpr auto get_var_type(T &&) noexcept { return get_type<T>(); }

  template <typename CharT, std::size_t N>
  auto convert_char_array_to_string(const std::array<CharT, N> &str) noexcept {
    return std::basic_string<CharT>{std::begin(str), std::end(str)};
  }
  template <typename T> auto get_var_type_str(T &&v) noexcept {
    return convert_char_array_to_string(get_var_type(FWD(v)));
  }
  template <typename T> auto get_type_str() noexcept {
    return convert_char_array_to_string(get_type<T>());
  }

}  // namespace zs
