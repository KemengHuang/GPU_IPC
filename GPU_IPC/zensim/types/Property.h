#pragma once
#include <type_traits>

#include "../meta/Meta.h"
#include "../meta/Relationship.h"

namespace zs {

  // HOST, DEVICE, UM
  enum struct memsrc_e : unsigned char { host = 0, device, um };
  using host_mem_tag = wrapv<memsrc_e::host>;
  using device_mem_tag = wrapv<memsrc_e::device>;
  using um_mem_tag = wrapv<memsrc_e::um>;
  constexpr auto mem_host = host_mem_tag{};
  constexpr auto mem_device = device_mem_tag{};
  constexpr auto mem_um = um_mem_tag{};

  template <typename Tag> constexpr bool is_memory_tag(Tag = {}) noexcept {
    return (
        is_same_v<Tag,
                  host_mem_tag> || is_same_v<Tag, device_mem_tag> || is_same_v<Tag, um_mem_tag>);
  }

  enum struct execspace_e : unsigned char { host = 0, openmp, cuda, hip };
  using host_exec_tag = wrapv<execspace_e::host>;
  using omp_exec_tag = wrapv<execspace_e::openmp>;
  using cuda_exec_tag = wrapv<execspace_e::cuda>;
  using hip_exec_tag = wrapv<execspace_e::hip>;
  constexpr auto exec_seq = host_exec_tag{};
  constexpr auto exec_omp = omp_exec_tag{};
  constexpr auto exec_cuda = cuda_exec_tag{};
  constexpr auto exec_hip = hip_exec_tag{};

  template <typename Tag> constexpr bool is_execution_tag(Tag = {}) noexcept {
    return (
        is_same_v<
            Tag,
            host_exec_tag> || is_same_v<Tag, omp_exec_tag> || is_same_v<Tag, cuda_exec_tag> || is_same_v<Tag, hip_exec_tag>);
  }

  enum struct attrib_e : unsigned char { scalar = 0, vector, matrix, affine };
  using attrib_scalar_tag = wrapv<attrib_e::scalar>;
  using attrib_vector_tag = wrapv<attrib_e::vector>;
  using attrib_matrix_tag = wrapv<attrib_e::matrix>;
  using attrib_affine_matrix_tag = wrapv<attrib_e::affine>;
  constexpr auto scalar_c = attrib_scalar_tag{};
  constexpr auto vector_c = attrib_vector_tag{};
  constexpr auto matrix_c = attrib_matrix_tag{};
  constexpr auto affine_matrix_c = attrib_affine_matrix_tag{};

  template <typename Tag> constexpr bool is_attribute_tag(Tag = {}) noexcept {
    return (
        is_same_v<
            Tag,
            attrib_scalar_tag> || is_same_v<Tag, attrib_vector_tag> || is_same_v<Tag, attrib_matrix_tag> || is_same_v<Tag, attrib_affine_matrix_tag>);
  }

  enum struct layout_e : int { aos = 0, soa, aosoa };
  using layout_aos_tag = wrapv<layout_e::aos>;
  using layout_soa_tag = wrapv<layout_e::soa>;
  using layout_aosoa_tag = wrapv<layout_e::aosoa>;
  constexpr auto aos_c = layout_aos_tag{};
  constexpr auto soa_c = layout_soa_tag{};
  constexpr auto aosoa_c = layout_aosoa_tag{};

  template <typename Tag> constexpr bool is_layout_tag(Tag = {}) noexcept {
    return (
        is_same_v<
            Tag,
            layout_aos_tag> || is_same_v<Tag, layout_soa_tag> || is_same_v<Tag, layout_aosoa_tag>);
  }

  enum struct kernel_e { linear = 2, quadratic = 3, cubic = 4 };
  using kernel_linear_tag = wrapv<kernel_e::linear>;
  using kernel_quadratic_tag = wrapv<kernel_e::quadratic>;
  using kernel_cubic_tag = wrapv<kernel_e::cubic>;
  constexpr auto kernel_linear_c = kernel_linear_tag{};
  constexpr auto kernel_quad_c = kernel_quadratic_tag{};
  constexpr auto kernel_cubic_c = kernel_cubic_tag{};

  template <typename Tag> constexpr bool is_kernel_tag(Tag = {}) noexcept {
    return (
        is_same_v<
            Tag,
            kernel_linear_tag> || is_same_v<Tag, kernel_quadratic_tag> || is_same_v<Tag, kernel_cubic_tag>);
  }

  enum struct grid_e : unsigned char { collocated = 0, cellcentered, staggered, total };
  using grid_collocated_tag = wrapv<grid_e::collocated>;
  using grid_cellcentered_tag = wrapv<grid_e::cellcentered>;
  using grid_staggered_tag = wrapv<grid_e::staggered>;
  static constexpr auto collocated_c = grid_collocated_tag{};
  static constexpr auto cellcentered_c = grid_cellcentered_tag{};
  static constexpr auto staggered_c = grid_staggered_tag{};

  template <typename Tag> constexpr bool is_grid_tag(Tag = {}) noexcept {
    return (
        is_same_v<
            Tag,
            grid_collocated_tag> || is_same_v<Tag, grid_cellcentered_tag> || is_same_v<Tag, grid_staggered_tag>);
  }

  /// comparable
  template <typename T> struct is_equality_comparable {
  private:
    static void *conv(bool);
    template <typename U> static std::true_type test(
        decltype(conv(std::declval<U const &>() == std::declval<U const &>())),
        decltype(conv(!std::declval<U const &>() == std::declval<U const &>())));
    template <typename U> static std::false_type test(...);

  public:
    static constexpr bool value = decltype(test<T>(nullptr, nullptr))::value;
  };

  template <class T> struct is_unbounded_array : std::false_type {};

  template <class T> struct is_unbounded_array<T[]> : std::true_type {};

}  // namespace zs
