// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
namespace mpl
{
// dummy tuple with low instantiation cost
template <typename... Tp>
struct type_list
{
    static constexpr auto size() { return sizeof...(Tp); }
};

/// get the index of a type in expansion
template <typename Tp, typename Type>
struct index_of;

template <typename Tp, template <typename...> class Tuple, typename... Types>
struct index_of<Tp, Tuple<Tp, Types...>>
{
    static constexpr size_t value = 0;
};

template <typename Tp, typename Head, template <typename...> class Tuple, typename... Tail>
struct index_of<Tp, Tuple<Head, Tail...>>
{
    static constexpr size_t value = 1 + index_of<Tp, Tuple<Tail...>>::value;
};

/// get the index of a type in expansion
template <typename Tp>
struct size_of;

template <typename... Tp>
struct size_of<type_list<Tp...>>
{
    static constexpr size_t value = sizeof...(Tp);
};

template <typename... Tp>
struct size_of<std::tuple<Tp...>>
{
    static constexpr size_t value = sizeof...(Tp);
};

// check if type is in expansion
//
template <typename... Tp>
struct is_one_of
{
    static constexpr bool value = false;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, S, Tuple<T...>>
{
    static constexpr bool value = std::is_same<F, S>::value || is_one_of<F, Tuple<T...>>::value;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, Tuple<S, T...>>
{
    static constexpr bool value = is_one_of<F, S, Tuple<T...>>::value;
};

template <typename Tp>
struct is_pair_impl
{
    static constexpr auto value = false;
};

template <typename LhsT, typename RhsT>
struct is_pair_impl<std::pair<LhsT, RhsT>>
{
    static constexpr auto value = true;
};

template <typename Tp>
struct is_pair : is_pair_impl<std::remove_cv_t<std::remove_reference_t<std::decay_t<Tp>>>>
{};

template <typename Tp>
struct is_string_type_impl
{
    static constexpr auto value =
        is_one_of<Tp, type_list<const char*, char*, std::string, std::string_view>>::value;
};

template <typename Tp>
struct is_string_type
: is_string_type_impl<std::remove_cv_t<std::remove_reference_t<std::decay_t<Tp>>>>
{};

template <typename, typename = void>
constexpr bool is_type_complete_v = false;  // NOLINT(misc-definitions-in-headers)

template <typename T>  // NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr bool is_type_complete_v<T, std::void_t<decltype(sizeof(T))>> = true;
}  // namespace mpl
}  // namespace common
}  // namespace rocprofiler
