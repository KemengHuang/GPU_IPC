#pragma once
#include <memory>
#include <type_traits>

namespace zs {

  /// this impl only accounts for single object
  template <class T, typename Deleter = std::default_delete<T>> class copyable_ptr {
  public:
    static_assert(!std::is_array_v<T>, "copyable_ptr does not support array");
    static_assert(std::is_copy_constructible_v<T>, "T should be copy-constructible");

    using pointer = std::add_pointer_t<T>;
    using element_type = T;
    using deleter_type = Deleter;

    template <typename U, typename E, template <typename, typename> class PTR>
    static constexpr bool __safe_conversion_up() noexcept {
      return std::is_convertible_v<typename PTR<U, E>::pointer, pointer> && !std::is_array_v<U>;
    }

    /// ctor
    copyable_ptr() = default;
    explicit copyable_ptr(T *p) noexcept : _ptr{p} {}

    /// deleter ctor mimic behavior of std::unique_ptr
    template <typename D = Deleter, std::enable_if_t<std::is_copy_constructible_v<D>, char> = 0>
    copyable_ptr(T *p, const Deleter &d) noexcept : _ptr{p, d} {}
    template <typename D = Deleter, std::enable_if_t<std::is_move_constructible_v<D>, char> = 0>
    copyable_ptr(T *p, std::enable_if_t<!std::is_lvalue_reference_v<D>, D &&> d) noexcept
        : _ptr{p, std::move(d)} {}
    template <typename D = Deleter, typename DUnref = std::remove_reference_t<D>>
    copyable_ptr(T *p, std::enable_if_t<std::is_lvalue_reference_v<D>, DUnref &&> d) = delete;

    constexpr copyable_ptr(std::nullptr_t) noexcept : _ptr{nullptr} {}

    /// override copy (assignment) ctor
    constexpr copyable_ptr(const copyable_ptr &o) noexcept(std::is_nothrow_copy_constructible_v<T>)
        : _ptr{std::unique_ptr<T, Deleter>(new T(*o), o.get_deleter())} {}
    copyable_ptr &operator=(const copyable_ptr &o) noexcept(
        std::is_nothrow_copy_constructible_v<T>) {
      _ptr = std::unique_ptr<T, Deleter>(new T(*o), o.get_deleter());
      return *this;
    }
    constexpr copyable_ptr(const std::unique_ptr<T, Deleter> &o) noexcept(
        std::is_nothrow_copy_constructible_v<T>)
        : _ptr{std::unique_ptr<T, Deleter>(new T(*o), o.get_deleter())} {}
    copyable_ptr &operator=(const std::unique_ptr<T, Deleter> &o) noexcept(
        std::is_nothrow_copy_constructible_v<T>) {
      _ptr = std::unique_ptr<T, Deleter>(new T(*o), o.get_deleter());
      return *this;
    }
    /// delegate move conversion (assignment) ctor
    template <typename U, typename E,
              std::enable_if_t<
                  __safe_conversion_up<U, E, copyable_ptr>()
                      && std::conditional_t<std::is_reference_v<Deleter>, std::is_same<E, Deleter>,
                                            std::is_convertible<E, Deleter>>::value,
                  int> = 0>
    copyable_ptr(copyable_ptr<U, E> &&o) noexcept : _ptr{std::move(o._ptr)} {}
    template <typename U, typename E,
              std::enable_if_t<
                  __safe_conversion_up<U, E, std::unique_ptr>()
                      && std::conditional_t<std::is_reference_v<Deleter>, std::is_same<E, Deleter>,
                                            std::is_convertible<E, Deleter>>::value,
                  int> = 0>
    copyable_ptr(std::unique_ptr<U, E> &&o) noexcept : _ptr{std::move(o)} {}

    template <typename U, typename E,
              std::enable_if_t<
                  __safe_conversion_up<U, E, copyable_ptr>()
                      && std::conditional_t<std::is_reference_v<Deleter>, std::is_same<E, Deleter>,
                                            std::is_convertible<E, Deleter>>::value,
                  int> = 0>
    copyable_ptr &operator=(copyable_ptr<U, E> &&o) noexcept {
      _ptr = std::move(o._ptr);
      return *this;
    }
    template <typename U, typename E,
              std::enable_if_t<
                  __safe_conversion_up<U, E, std::unique_ptr>()
                      && std::conditional_t<std::is_reference_v<Deleter>, std::is_same<E, Deleter>,
                                            std::is_convertible<E, Deleter>>::value,
                  int> = 0>
    copyable_ptr &operator=(std::unique_ptr<U, E> &&o) noexcept {
      _ptr = std::move(o);
      return *this;
    }

    ~copyable_ptr() = default;

    copyable_ptr(copyable_ptr &&o) = default;
    copyable_ptr &operator=(copyable_ptr &&o) = default;

    /// observer delegation
    std::add_lvalue_reference_t<element_type> operator*() const { return *_ptr; }
    pointer operator->() const noexcept { return _ptr.operator->(); }
    pointer get() const noexcept { return _ptr.get(); }
    deleter_type &get_deleter() noexcept { return _ptr.get_deleter(); }
    const deleter_type &get_deleter() const noexcept { return _ptr.get_deleter(); }
    explicit operator bool() const noexcept { return static_cast<bool>(_ptr); }

    /// modifier delegation
    pointer release() noexcept { return _ptr.release(); }
    void reset(pointer p = pointer()) noexcept { _ptr.reset(std::move(p)); }
    void swap(std::unique_ptr<T, Deleter> &o) noexcept { _ptr.swap(o); }
    void swap(copyable_ptr &o) noexcept { _ptr.swap(o._ptr); }

    operator std::unique_ptr<T, Deleter> &() noexcept { return _ptr; }
    operator const std::unique_ptr<T, Deleter> &() const noexcept { return _ptr; }

    std::unique_ptr<T, Deleter> _ptr;
  };
}  // namespace zs