#pragma once
#include "Meta.h"

namespace zs {

  // enable_if
  // conditional
  template <bool B> struct conditional_impl { template <class T, class F> using type = T; };
  template <> struct conditional_impl<false> { template <class T, class F> using type = F; };

  template <bool B, class T, class F> using conditional_t =
      typename conditional_impl<B>::template type<T, F>;

}  // namespace zs
