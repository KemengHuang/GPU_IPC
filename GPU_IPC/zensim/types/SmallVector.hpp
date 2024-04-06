#pragma once
#include <string>
#include <string_view>

#include "../meta/Meta.h"

namespace zs {

  // null-terminated string
  struct SmallString {
    using char_type = char;
    static_assert(std::is_trivial_v<char_type> && std::is_standard_layout_v<char_type>,
                  "char type is not trivial and in standard-layout.");
    using size_type = std::size_t;
    static constexpr auto nbytes = 4 * sizeof(void *);  ///< 4 * 8 - 1 = 31 bytes (chars)

    constexpr SmallString() noexcept : buf{} {
      for (auto &c : buf) c = '\0';
    }
    constexpr SmallString(const char tmp[]) : buf{} {
      size_type i = 0;
      for (; i + (size_type)1 != nbytes && tmp[i]; ++i) buf[i] = tmp[i];
      buf[i] = '\0';
    }
    SmallString(const std::string &str) noexcept {
      size_type n = str.size() < nbytes ? str.size() : nbytes - 1;
      buf[n] = '\0';
      for (size_type i = 0; i != n; ++i) buf[i] = str[i];
    }
    constexpr SmallString(const SmallString &) noexcept = default;
    constexpr SmallString &operator=(const SmallString &) noexcept = default;
    constexpr SmallString(SmallString &&) noexcept = default;
    constexpr SmallString &operator=(SmallString &&) noexcept = default;

    constexpr bool operator==(const char str[]) const noexcept {
      size_type i = 0;
      for (; buf[i] && str[i]; ++i)
        if (buf[i] != str[i]) return false;
      if (!(buf[i] || str[i])) return true;
      return false;
    }
    constexpr bool operator==(const std::string_view str) const noexcept {
      size_type i = 0;
      for (; buf[i] && i != str.size(); ++i)
        if (buf[i] != str[i]) return false;
      if (!(buf[i] || str[i])) return true;
      return false;
    }

    std::string asString() const { return std::string{buf}; }
    constexpr const char *asChars() const noexcept { return buf; }
    constexpr operator const char *() const noexcept { return buf; }
    constexpr size_type size() const noexcept {
      size_type i = 0;
      for (; buf[i]; ++i)
        ;
      return i;
    }

    alignas(nbytes) char_type buf[nbytes];
  };

  /// property tag
  struct PropertyTag {
    SmallString name;
    int numChannels;
  };

}  // namespace zs