#pragma once
#include "QRSVD.hpp"

namespace zs::math {

  template <typename VecTM,
            enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value
                                               == VecTM::template range_t<1>::value> = 0>
  constexpr auto decompose_transform(const VecInterface<VecTM> &m,
                                     bool applyOnColumn = true) noexcept {
    constexpr auto dim = VecTM::template range_t<0>::value - 1;
    static_assert(VecTM::template range_t<0>::value <= 4 && (VecTM::template range_t<0>::value > 1),
                  "transformation should be of 2x2, 3x3 or 4x4 shape only.");
    using Mat = decltype(m.clone());
    using ValT = typename VecTM::value_type;
    using Tn = typename VecTM::index_type;
    auto H = m.clone();
    if (!applyOnColumn) H = H.transpose();
    // T
    auto T = Mat::identity();
    for (Tn i = 0; i != dim; ++i) {
      T(i, dim) = H(i, dim);
      H(i, dim) = 0;
    }
    // RS
    typename VecTM::template variant_vec<ValT, integer_seq<Tn, dim, dim>> L{};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) L(i, j) = H(i, j);
    auto [R_, S_] = polar_decomposition(L);
    auto R{Mat::zeros()}, S{Mat::zeros()};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) {
        R(i, j) = R_(i, j);
        S(i, j) = S_(i, j);
      }
    R(dim, dim) = S(dim, dim) = (ValT)1;
    if (applyOnColumn) return std::make_tuple(T, R, S);
    return std::make_tuple(S.transpose(), R.transpose(), T.transpose());
  }

  template <
      typename VecTM, typename VecTS, typename VecTR, typename VecTT,
      enable_if_all<VecTM::dim == 2,
                    VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                    VecTS::dim == 2,
                    VecTS::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    VecTS::template range_t<1>::value + 1 == VecTM::template range_t<0>::value,
                    VecTR::dim == 2,
                    VecTR::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    VecTR::template range_t<1>::value + 1 == VecTM::template range_t<0>::value,
                    VecTT::dim == 1,
                    VecTT::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    std::is_floating_point_v<typename VecTM::value_type>,
                    std::is_floating_point_v<typename VecTS::value_type>,
                    std::is_floating_point_v<typename VecTR::value_type>,
                    std::is_floating_point_v<typename VecTT::value_type>> = 0>
  constexpr void decompose_transform(const VecInterface<VecTM> &m, VecInterface<VecTS> &s,
                                     VecInterface<VecTR> &r, VecInterface<VecTT> &t,
                                     bool applyOnColumn = true) noexcept {
    constexpr auto dim = VecTM::template range_t<0>::value - 1;
    static_assert(VecTM::template range_t<0>::value <= 4 && (VecTM::template range_t<0>::value > 1),
                  "transformation should be of 2x2, 3x3 or 4x4 shape only.");
    using ValT = typename VecTM::value_type;
    using Tn = typename VecTM::index_type;
    auto H = m.clone();
    if (!applyOnColumn) H = H.transpose();
    // T
    for (Tn i = 0; i != dim; ++i) t(i) = H(i, dim);
    // RS
    typename VecTM::template variant_vec<ValT, integer_seq<Tn, dim, dim>> L{};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) L(i, j) = H(i, j);
    auto [R_, S_] = polar_decomposition(L);

    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) {
        r(i, j) = R_(i, j);
        s(i, j) = S_(i, j);
      }
  }

}  // namespace zs::math