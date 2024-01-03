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

#include "lib/rocprofiler-sdk/buffer.hpp"

#include <glog/logging.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <random>
#include <vector>

namespace rocprofiler
{
namespace buffer
{
namespace
{
using reserve_size_t = common::container::reserve_size;

auto&
get_buffers_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

uint64_t
get_buffer_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}
}  // namespace

bool
is_valid_buffer_id(rocprofiler_buffer_id_t id)
{
    auto nbuffers = get_buffers().size();
    auto offset   = get_buffer_offset();
    return (id.handle >= offset && id.handle < (offset + nbuffers));
}

unique_buffer_vec_t&
get_buffers()
{
    static auto _v = unique_buffer_vec_t{reserve_size_t{unique_buffer_vec_t::chunk_size}};
    return _v;
}

instance*
get_buffer(rocprofiler_buffer_id_t buffer_id)
{
    if(is_valid_buffer_id(buffer_id))
    {
        for(auto& itr : get_buffers())
        {
            if(itr && itr->buffer_id == buffer_id.handle)
            {
                return itr.get();
            }
        }
    }
    return nullptr;
}

std::optional<rocprofiler_buffer_id_t>
allocate_buffer()
{
    // ensure buffer has thread to handle flushing it
    static auto _init_threads_once = std::once_flag{};
    std::call_once(_init_threads_once, []() { internal_threading::initialize(); });

    // ... allocate any internal space needed to handle another context ...
    auto _lk = std::unique_lock<std::mutex>{get_buffers_mutex()};

    // initial context identifier number
    auto _idx = get_buffer_offset() + get_buffers().size();

    // make space in registered
    get_buffers().emplace_back(nullptr);

    // create an entry in the registered
    auto& _cfg_v = get_buffers().back();
    _cfg_v       = allocator::make_unique_static<buffer::instance>();
    auto* _cfg   = _cfg_v.get();

    if(!_cfg) return std::nullopt;

    // set the buffer id value
    _cfg_v->buffer_id = _idx;

    return rocprofiler_buffer_id_t{_idx};
}

rocprofiler_status_t
flush(rocprofiler_buffer_id_t buffer_id, bool wait)
{
    if(registration::get_fini_status() > 0) return ROCPROFILER_STATUS_SUCCESS;

    auto offset = get_buffer_offset();

    if(!is_valid_buffer_id(buffer_id)) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto* buff = get_buffer(buffer_id);

    auto* task_group =
        internal_threading::get_task_group(rocprofiler_callback_thread_t{buff->task_group_id});

    if(wait && task_group) task_group->wait();

    // buffer is currently being flushed or destroyed
    if(buff->syncer.test_and_set())
    {
        if(!wait) return ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;
        while(buff->syncer.test_and_set())
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
    }

    auto idx = buff->buffer_idx++;

    auto _task = [buffer_id, idx, offset]() {
        auto& buff_v          = get_buffers().at(buffer_id.handle - offset);
        auto& buff_internal_v = buff_v->get_internal_buffer(idx);

        if(!buff_internal_v.is_empty())
        {
            // get the array of record headers
            auto buff_data = buff_internal_v.get_record_headers();

            // invoke buffer callback
            try
            {
                if(buff_v->callback)
                {
                    buff_v->callback(rocprofiler_context_id_t{buff_v->context_id},
                                     rocprofiler_buffer_id_t{buff_v->buffer_id},
                                     buff_data.data(),
                                     buff_data.size(),
                                     buff_v->callback_data,
                                     buff_v->drop_count);
                }
            } catch(std::exception& e)
            {
                LOG(ERROR) << "buffer callback threw an exception: " << e.what();
            }
            // clear the buffer
            buff_internal_v.clear();
        }
        else
        {
            LOG(INFO) << "buffer at " << buffer_id.handle << " is empty...";
        }

        buff_v->syncer.clear();
    };

    if(task_group)
    {
        task_group->exec(_task);
        if(wait) task_group->join();
    }
    else
    {
        _task();
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace buffer
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                          size_t                          size,
                          size_t                          watermark,
                          rocprofiler_buffer_policy_t     action,
                          rocprofiler_buffer_tracing_cb_t callback,
                          void*                           callback_data,
                          rocprofiler_buffer_id_t*        buffer_id)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* existing_buff = rocprofiler::buffer::get_buffer(*buffer_id);
    if(existing_buff)
    {
        LOG(ERROR) << "buffer (handle=" << buffer_id->handle
                   << ") already allocated: handle=" << existing_buff->buffer_id;
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    }

    auto opt_buff_id = rocprofiler::buffer::allocate_buffer();
    if(!opt_buff_id) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
    buffer_id->handle = opt_buff_id->handle;

    auto& buff = rocprofiler::buffer::get_buffers().at(opt_buff_id->handle -
                                                       rocprofiler::buffer::get_buffer_offset());

    // allocate the buffers. if it is lossless, we allocate a second buffer to store data while
    // other buffer is being flushed
    buff->buffers.front().allocate(size);
    if(action == ROCPROFILER_BUFFER_POLICY_LOSSLESS) buff->buffers.back().allocate(size);

    buff->watermark     = watermark;
    buff->policy        = action;
    buff->callback      = callback;
    buff->callback_data = callback_data;
    buff->context_id    = context.handle;
    buff->buffer_id     = buffer_id->handle;
    buff->buffer_idx    = 0;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id)
{
    return rocprofiler::buffer::flush(buffer_id, true);
}

rocprofiler_status_t
rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id)
{
    if(!rocprofiler::buffer::is_valid_buffer_id(buffer_id))
        return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto  offset  = rocprofiler::buffer::get_buffer_offset();
    auto& buffers = rocprofiler::buffer::get_buffers();
    auto& buff    = buffers.at(buffer_id.handle - offset);

    // buffer is currently being flushed or destroyed
    if(buff->syncer.test_and_set()) return ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;

    for(auto& itr : buff->buffers)
        itr.reset();

    buff->syncer.clear();

    return ROCPROFILER_STATUS_SUCCESS;
}
}
