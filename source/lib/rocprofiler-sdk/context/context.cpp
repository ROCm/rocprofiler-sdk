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

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"

#include <glog/logging.h>

#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>

namespace rocprofiler
{
namespace context
{
namespace
{
using reserve_size_t = common::container::reserve_size;
using unique_context_vec_t =
    common::container::stable_vector<allocator::unique_static_ptr_t<context>, 8>;
using active_context_vec_t = common::container::stable_vector<std::atomic<const context*>, 8>;

constexpr auto invalid_client_idx = std::numeric_limits<uint32_t>::max();

auto&
get_contexts_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

uint64_t
get_contexts_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}

auto&
get_client_index()
{
    static auto _v = invalid_client_idx;
    return _v;
}

unique_context_vec_t&
get_registered_contexts_impl()
{
    static auto _v = unique_context_vec_t{reserve_size_t{unique_context_vec_t::chunk_size}};
    return _v;
}

auto&
get_num_active_contexts()
{
    static auto _v = std::atomic<int64_t>{0};
    return _v;
}

active_context_vec_t&
get_active_contexts_impl()
{
    static auto* _v    = new active_context_vec_t{reserve_size_t{active_context_vec_t::chunk_size}};
    static auto  _once = std::once_flag{};
    std::call_once(_once, std::atexit, []() {
        for(auto& itr : *_v)
        {
            itr.store(nullptr);
        }
    });
    return *_v;
}

auto*&
get_correlation_id_map()
{
    using data_type  = common::container::stable_vector<std::unique_ptr<correlation_id>>;
    static auto*& _v = common::static_object<common::Synchronized<data_type>>::construct();
    return _v;
}

auto&
get_latest_correlation_id_impl()
{
    static thread_local auto _v = common::container::small_vector<correlation_id*, 16>{};
    return _v;
}

uint64_t
get_unique_internal_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return ++_v;
}
}  // namespace

uint32_t
correlation_id::add_ref_count()
{
    return m_ref_count.fetch_add(1);
}

uint32_t
correlation_id::sub_ref_count()
{
    auto _ret = m_ref_count.fetch_sub(1);

    LOG_IF(FATAL, _ret == 0) << "correlation id underflow";

    if(_ret == 1)
    {
        auto ctxs = get_active_contexts([](const context* ctx) {
            return (ctx->buffered_tracer &&
                    (ctx->buffered_tracer->domains(
                        ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)));
        });

        auto record = rocprofiler_buffer_tracing_correlation_id_retirement_record_t{
            .size      = sizeof(rocprofiler_buffer_tracing_correlation_id_retirement_record_t),
            .kind      = ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
            .timestamp = common::timestamp_ns(),
            .internal_correlation_id = internal};

        if(!ctxs.empty())
        {
            for(const auto* itr : ctxs)
            {
                auto* _buffer = buffer::get_buffer(itr->buffered_tracer->buffer_data.at(
                    ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT));

                auto success = CHECK_NOTNULL(_buffer)->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_TRACING,
                    ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
                    record);

                LOG_IF(FATAL, !success) << "failed to emplace correlation id retirement";
            }
        }
    }

    return _ret;
}

uint32_t
correlation_id::add_kern_count()
{
    return m_kern_count.fetch_add(1);
}

uint32_t
correlation_id::sub_kern_count()
{
    return m_kern_count.fetch_sub(1);
}

correlation_id*
correlation_tracing_service::construct(uint32_t _init_ref_count)
{
    auto  _internal_id = get_unique_internal_id();
    auto* corr_id_map  = get_correlation_id_map();
    if(!corr_id_map) return nullptr;
    auto& ret = corr_id_map->wlock([](auto& data) -> auto& { return data.emplace_back(); });
    ret       = std::make_unique<correlation_id>(_init_ref_count, common::get_tid(), _internal_id);

    get_latest_correlation_id_impl().emplace_back(ret.get());

    return ret.get();
}

correlation_id*
get_latest_correlation_id()
{
    return (get_latest_correlation_id_impl().empty()) ? nullptr
                                                      : get_latest_correlation_id_impl().back();
}

const correlation_id*
pop_latest_correlation_id(correlation_id* val)
{
    if(!val)
    {
        LOG(ERROR) << "passed nullptr to correlation id";
        return nullptr;
    }

    if(get_latest_correlation_id_impl().empty())
    {
        LOG(ERROR) << "empty thread-local correlation id stack";
        return nullptr;
    }

    LOG_IF(ERROR, get_latest_correlation_id_impl().back() != val)
        << "pop_latest_correlation_id is happening out of order for " << val->internal
        << ". top of stack is " << get_latest_correlation_id_impl().back()->internal;

    get_latest_correlation_id_impl().pop_back();

    return (get_latest_correlation_id_impl().empty()) ? nullptr
                                                      : get_latest_correlation_id_impl().back();
}

context_array_t&
get_registered_contexts(context_array_t& data, context_filter_t filter)
{
    data.clear();
    auto num_ctx = get_registered_contexts_impl().size();
    if(num_ctx <= 0) return data;

    data.reserve(num_ctx);
    for(auto& itr : get_registered_contexts_impl())
    {
        const auto* ctx = itr.get();
        if(ctx)
        {
            if(!filter || (filter && filter(ctx))) data.emplace_back(ctx);
        }
    }
    return data;
}

context_array_t
get_registered_contexts(context_filter_t filter)
{
    auto data = context_array_t{};
    get_registered_contexts(data, filter);
    return data;
}

context_array_t&
get_active_contexts(context_array_t& data, context_filter_t filter)
{
    data.clear();
    auto num_ctx = get_num_active_contexts().load(std::memory_order_acquire);
    if(num_ctx <= 0) return data;

    data.reserve(num_ctx);
    for(auto& itr : get_active_contexts_impl())
    {
        const auto* ctx = itr.load(std::memory_order_acquire);
        if(ctx)
        {
            if(!filter || (filter && filter(ctx))) data.emplace_back(ctx);
        }
        if(static_cast<int64_t>(data.size()) == num_ctx)
        {
            // if the number of active contexts changed, restart
            if(num_ctx != get_num_active_contexts().load(std::memory_order_relaxed))
            {
                data.clear();
                return get_active_contexts(data, filter);
            }
            break;
        }
    }
    return data;
}

context_array_t
get_active_contexts(context_filter_t filter)
{
    auto data = context_array_t{};
    get_active_contexts(data, filter);
    return data;
}

// set the client index needs to be called before allocate_context()
void
push_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == invalid_client_idx)
        << " rocprofiler client index is currently " << get_client_index()
        << "... which means that a new client is initializing before the last client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = value;
}

// remove the client index
void
pop_client(uint32_t value)
{
    LOG_ASSERT(get_client_index() == value)
        << " rocprofiler client index is currently not " << value
        << "... which means that a new client was initialized before this client finished "
           "initializing. This is an internal error, please file a bug report with a reproducer";
    get_client_index() = invalid_client_idx;
}

std::optional<rocprofiler_context_id_t>
allocate_context()
{
    // ... allocate any internal space needed to handle another context ...
    auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

    // initial context identifier number
    auto _idx = get_registered_contexts_impl().size() + get_contexts_offset();

    // make space in registered
    get_registered_contexts_impl().emplace_back(nullptr);

    // create an entry in the registered
    auto& _cfg_v = get_registered_contexts_impl().back();
    _cfg_v       = allocator::make_unique_static<context>();
    auto* _cfg   = _cfg_v.get();
    // ...

    if(!_cfg) return std::nullopt;

    _cfg->size        = sizeof(context);
    _cfg->context_idx = _idx;
    _cfg->client_idx  = get_client_index();

    LOG_ASSERT(_cfg->client_idx != invalid_client_idx)
        << " rocprofiler internal error: a context was allocated without an associated tool client "
           "identifier";

    return rocprofiler_context_id_t{_idx};
}

context*
get_mutable_registered_context(rocprofiler_context_id_t id)
{
    if(id.handle < get_contexts_offset()) return nullptr;
    auto _idx = id.handle - get_contexts_offset();
    if(_idx >= get_registered_contexts_impl().size()) return nullptr;
    return get_registered_contexts_impl().at(_idx).get();
}

const context*
get_registered_context(rocprofiler_context_id_t id)
{
    return get_mutable_registered_context(id);
}

rocprofiler_status_t
validate_context(const context* cfg)
{
    return (cfg) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
}

rocprofiler_status_t
start_context(rocprofiler_context_id_t context_id)
{
    if(context_id.handle < get_contexts_offset()) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if((context_id.handle - get_contexts_offset()) >= get_registered_contexts_impl().size())
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    const auto* cfg = get_registered_context(context_id);

    if(!cfg) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(validate_context(cfg) != ROCPROFILER_STATUS_SUCCESS)
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto current_contexts = context_array_t{};
    for(const auto* itr : get_active_contexts(current_contexts))
    {
        if(cfg->context_idx == itr->context_idx)
        {
            return ROCPROFILER_STATUS_SUCCESS;
        }
        else if(cfg->counter_collection && itr->counter_collection)
        {
            // conflicting context
            return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
        }
    }

    uint64_t rocp_tot_contexts = get_registered_contexts_impl().size();
    auto     idx               = rocp_tot_contexts;
    auto&    active_contexts   = get_active_contexts_impl();
    {
        // hold a lock here to prevent multiple threads from finding the same nullptr slot
        auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};
        // try to find a nullptr slot first
        for(size_t i = 0; i < active_contexts.size(); ++i)
        {
            const auto* itr = active_contexts.at(i).load(std::memory_order_relaxed);
            if(itr == nullptr)
            {
                idx = i;
                break;
            }
            else if(context_id.handle == itr->context_idx)
            {
                return ROCPROFILER_STATUS_SUCCESS;
            }
        }
        // if no nullptr slot was found, then create one while lock is held
        if(idx == rocp_tot_contexts)
        {
            idx = active_contexts.size();
            active_contexts.emplace_back();
        }

        get_num_active_contexts().fetch_add(1, std::memory_order_release);
    }

    // atomic swap the pointer into the "active" array used internally
    const context* _expected = nullptr;
    bool           success   = active_contexts.at(idx).compare_exchange_strong(
        _expected, get_registered_context(context_id));

    if(!success)
    {
        get_num_active_contexts().fetch_sub(1, std::memory_order_release);
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    }

    if(cfg->counter_collection) rocprofiler::counters::start_context(cfg);

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
stop_context(rocprofiler_context_id_t idx)
{
    // hold a lock here to prevent other thread from changing the active contexts array
    auto _lk = std::unique_lock<std::mutex>{get_contexts_mutex()};

    // atomically assign the context pointer to NULL so that it is skipped in future
    // callbacks
    for(auto& itr : get_active_contexts_impl())
    {
        const context* _expected = itr.load(std::memory_order_acquire);
        if(_expected && _expected->context_idx == idx.handle)
        {
            bool success = itr.compare_exchange_strong(_expected, nullptr);

            if(success)
            {
                auto nactive = get_num_active_contexts().load(std::memory_order_acquire);
                if(nactive > 0) get_num_active_contexts().fetch_sub(1, std::memory_order_release);

                if(_expected->counter_collection)
                    rocprofiler::counters::stop_context(const_cast<context*>(_expected));
                return ROCPROFILER_STATUS_SUCCESS;
            }
        }
    }

    return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;  // compare exchange failed
}

void
deactivate_client_contexts(rocprofiler_client_id_t client_id)
{
    for(auto& itr : get_active_contexts_impl())
    {
        const auto* itr_v = itr.load();
        if(itr_v && itr_v->client_idx == client_id.handle)
        {
            itr.store(nullptr);
        }
    }
}

void
deregister_client_contexts(rocprofiler_client_id_t client_id)
{
    for(auto& itr : get_registered_contexts_impl())
    {
        if(itr->client_idx == client_id.handle && buffer::get_buffers())
        {
            for(auto& bitr : *buffer::get_buffers())
            {
                if(bitr && bitr->context_id == itr->context_idx) bitr.reset();
            }
            itr.reset();
        }
    }
}
}  // namespace context
}  // namespace rocprofiler
