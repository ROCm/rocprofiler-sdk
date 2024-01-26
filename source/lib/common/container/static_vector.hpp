// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "lib/common/container/c_array.hpp"
#include "lib/common/defines.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename Tp, size_t N, bool AtomicSizeV = false>
struct static_vector
{
    using count_type = std::conditional_t<AtomicSizeV, std::atomic<size_t>, size_t>;
    using this_type  = static_vector<Tp, N>;
    using value_type = Tp;

    static_vector()                         = default;
    static_vector(const static_vector&)     = default;
    static_vector(static_vector&&) noexcept = default;
    static_vector& operator=(const static_vector&) = default;
    static_vector& operator=(static_vector&&) noexcept = default;

    explicit static_vector(size_t _n, Tp _v = {});
    explicit static_vector(c_array<Tp>&&);

    template <size_t M>
    explicit static_vector(std::array<Tp, M>&&);

    static_vector& operator=(std::initializer_list<Tp>&& _v);
    static_vector& operator=(std::pair<std::array<Tp, N>, size_t>&&);

    template <typename... Args>
    value_type& emplace_back(Args&&... _v);

    template <typename Up>
    decltype(auto) push_back(Up&& _v)
    {
        return emplace_back(Tp{std::forward<Up>(_v)});
    }

    void pop_back() { --m_size; }

    void clear();
    void reserve(size_t) noexcept {}
    void shrink_to_fit() noexcept {}
    auto capacity() noexcept { return N; }

    size_t size() const { return m_size; }
    bool   empty() const { return (size() == 0); }

    auto begin() { return m_data.begin(); }
    auto begin() const { return m_data.begin(); }
    auto cbegin() const { return m_data.cbegin(); }

    auto end() { return m_data.begin() + size(); }
    auto end() const { return m_data.begin() + size(); }
    auto cend() const { return m_data.cbegin() + size(); }

    decltype(auto) operator[](size_t _idx) { return m_data[_idx]; }
    decltype(auto) operator[](size_t _idx) const { return m_data[_idx]; }

    decltype(auto) at(size_t _idx) { return m_data.at(_idx); }
    decltype(auto) at(size_t _idx) const { return m_data.at(_idx); }

    decltype(auto) front() { return m_data.front(); }
    decltype(auto) front() const { return m_data.front(); }
    decltype(auto) back() { return *(m_data.begin() + size() - 1); }
    decltype(auto) back() const { return *(m_data.begin() + size() - 1); }

    auto*       data() { return m_data.data(); }
    const auto* data() const { return m_data.data(); }

    void swap(this_type& _v) noexcept;

    friend void swap(this_type& _lhs, this_type& _rhs) noexcept { _lhs.swap(_rhs); }

private:
    void update_size(size_t);

private:
    count_type        m_size = count_type{0};
    std::array<Tp, N> m_data = {};
};

template <typename Tp, size_t N, bool AtomicSizeV>
static_vector<Tp, N, AtomicSizeV>::static_vector(size_t _n, Tp _v)
{
    m_data.fill(_v);
    update_size(_n);
}

template <typename Tp, size_t N, bool AtomicSizeV>
static_vector<Tp, N, AtomicSizeV>::static_vector(c_array<Tp>&& _v)
{
    auto _n = std::min<size_t>(N, _v.size());
    for(size_t i = 0; i < _n; ++i, ++m_size)
        m_data[i] = _v[i];
}

template <typename Tp, size_t N, bool AtomicSizeV>
template <size_t M>
static_vector<Tp, N, AtomicSizeV>::static_vector(std::array<Tp, M>&& _v)
{
    auto _n = std::min<size_t>(N, M);
    for(size_t i = 0; i < _n; ++i, ++m_size)
        m_data[i] = _v[i];
}

template <typename Tp, size_t N, bool AtomicSizeV>
static_vector<Tp, N, AtomicSizeV>&
static_vector<Tp, N, AtomicSizeV>::operator=(std::initializer_list<Tp>&& _v)
{
    if(ROCPROFILER_UNLIKELY(_v.size() > N))
    {
        throw std::out_of_range(std::string{"static_vector::operator=(initializer_list) size > "} +
                                std::to_string(N));
    }

    clear();
    for(auto&& itr : _v)
        m_data[m_size++] = itr;
    return *this;
}

template <typename Tp, size_t N, bool AtomicSizeV>
static_vector<Tp, N, AtomicSizeV>&
static_vector<Tp, N, AtomicSizeV>::operator=(std::pair<std::array<Tp, N>, size_t>&& _v)
{
    update_size(0);
    m_data = std::move(_v.first);
    update_size(_v.second);

    return *this;
}

template <typename Tp, size_t N, bool AtomicSizeV>
void
static_vector<Tp, N, AtomicSizeV>::clear()
{
    update_size(0);
}

template <typename Tp, size_t N, bool AtomicSizeV>
void
static_vector<Tp, N, AtomicSizeV>::swap(this_type& _v) noexcept
{
    if constexpr(AtomicSizeV)
    {
        auto _t_size = m_size;
        auto _v_size = _v.m_size;
        std::swap(m_data, _v.m_data);
        update_size(_v_size);
        _v.update_size(_t_size);
    }
    else
    {
        std::swap(m_size, _v.m_size);
        std::swap(m_data, _v.m_data);
    }
}

template <typename Tp, size_t N, bool AtomicSizeV>
template <typename... Args>
Tp&
static_vector<Tp, N, AtomicSizeV>::emplace_back(Args&&... _v)
{
    auto _idx = m_size++;
    if(_idx >= N)
    {
        throw std::out_of_range(std::string{"static_vector::emplace_back - reached capacity "} +
                                std::to_string(N));
    }

    if constexpr(sizeof...(Args) > 0)
    {
        if constexpr(std::is_assignable<Tp, decltype(std::forward<Args>(_v))...>::value)
            m_data[_idx] = {std::forward<Args>(_v)...};
        else
            m_data[_idx] = Tp{std::forward<Args>(_v)...};
    }
    else
    {
        m_data[_idx] = {};
    }
    return m_data[_idx];
}

template <typename Tp, size_t N, bool AtomicSizeV>
void
static_vector<Tp, N, AtomicSizeV>::update_size(size_t _n)
{
    if constexpr(AtomicSizeV)
        m_size.store(_n);
    else
        m_size = _n;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
